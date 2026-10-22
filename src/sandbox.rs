//! Sandboxed probe execution.
//!
//! The Python reference never runs solver-written code in-process: a proposal is
//! materialised into a throwaway directory, executed there by an external agent
//! runner, and read back as files (`recursion.py::_roll`). This module is the
//! same shape for deterministic probes: a `HarnessState` becomes a temporary
//! workspace, and one script inside it is executed under a wall-clock budget.
//!
//! # Isolation boundary
//!
//! * Files are written under a private `tempfile::TempDir` (mode 0700 on Unix)
//!   that is deleted when the workspace is dropped.
//! * Relative paths are normalised with the same rule as the validation layer,
//!   so `..`, absolute paths, and duplicates are rejected before any write.
//! * Execution is delegated to [`crate::execution::Isolation`]. Production uses
//!   the macOS `sandbox-exec` backend and **fails closed** when it is
//!   unavailable; there is no host-Python fallback. The `cfg(test)` host backend
//!   exists only to exercise plumbing and does not claim isolation.
//!
//! # Budgets
//!
//! The wall-clock budget, process kill, pipe draining, environment clearing and
//! (for the sandbox backend) resource limits live in [`crate::execution`]. A
//! timeout is reported as [`SandboxError::Timeout`].

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tempfile::TempDir;

use crate::execution::{CancellationToken, ExecError, ExecRequest, Executor, Isolation, OutputStream, PartialOutput, DEFAULT_PYTHON};
use crate::validation::normalize_rel;


// ---------------------------------------------------------------------------
// Path validation
// ---------------------------------------------------------------------------

/// Why a candidate tree cannot be materialised.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreePathError {
    /// Absolute path, `..` escape, empty path, or otherwise non-normalisable.
    UnsafePath { path: PathBuf, reason: &'static str },
    /// Two keys normalise onto the same relative path.
    DuplicatePath { path: PathBuf, previous: PathBuf },
    /// Two keys differ only by case and would collide on a case-insensitive
    /// filesystem.
    CaseAlias { path: PathBuf, previous: PathBuf },
    /// A path component is (or is reached through) a symlink.
    Symlink { path: PathBuf },
}

impl std::fmt::Display for TreePathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsafePath { path, reason } => {
                write!(f, "unsafe harness path {}: {reason}", path.display())
            }
            Self::DuplicatePath { path, previous } => write!(
                f,
                "duplicate harness path {} (already provided as {})",
                path.display(),
                previous.display()
            ),
            Self::CaseAlias { path, previous } => write!(
                f,
                "case-aliased harness path {} (collides with {} on a case-insensitive filesystem)",
                path.display(),
                previous.display()
            ),
            Self::Symlink { path } => write!(
                f,
                "refusing to follow symlink while materialising {}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for TreePathError {}

/// Validate every key without touching the filesystem.
///
/// Rejects non-normalisable paths, duplicate normalised paths, and case aliases.
pub fn validate_tree_paths(tree: &BTreeMap<PathBuf, String>) -> Result<(), TreePathError> {
    let mut normalized: BTreeMap<String, PathBuf> = BTreeMap::new();
    let mut folded: BTreeMap<String, PathBuf> = BTreeMap::new();

    for path in tree.keys() {
        let relative = normalize_rel(path).ok_or_else(|| TreePathError::UnsafePath {
            path: path.clone(),
            reason: "absolute, empty, or escapes the harness root",
        })?;
        if let Some(previous) = normalized.insert(relative.clone(), path.clone()) {
            return Err(TreePathError::DuplicatePath {
                path: path.clone(),
                previous,
            });
        }
        let lower = relative.to_ascii_lowercase();
        if let Some(previous) = folded.insert(lower, path.clone()) {
            return Err(TreePathError::CaseAlias {
                path: path.clone(),
                previous,
            });
        }
    }
    Ok(())
}

/// Validate the tree and reject any destination reached through a symlink.
///
/// The workspace is created empty, so this is defence in depth: it protects
/// callers that materialise into a directory they did not create themselves.
pub fn validate_tree_against_root(
    root: &Path,
    tree: &BTreeMap<PathBuf, String>,
) -> Result<(), TreePathError> {
    validate_tree_paths(tree)?;
    for path in tree.keys() {
        let Some(relative) = normalize_rel(path) else {
            continue;
        };
        let mut current = root.to_path_buf();
        for part in relative.split('/') {
            current.push(part);
            if let Ok(metadata) = std::fs::symlink_metadata(&current) {
                if metadata.file_type().is_symlink() {
                    return Err(TreePathError::Symlink { path: current });
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

/// A materialised harness tree with a lifetime tied to one probe batch.
#[derive(Debug)]
pub struct SandboxWorkspace {
    dir: TempDir,
    executor: Option<Arc<dyn Executor>>,
    cancel: Option<CancellationToken>,
}

impl SandboxWorkspace {
    /// Write every entry of `state` into a fresh temporary directory.
    ///
    /// Parent directories are created on demand. Paths are normalised with the
    /// validation rule; an entry that escapes the root, or a second entry that
    /// normalises onto the same path, fails before anything is written.
    pub fn create(state: &BTreeMap<PathBuf, String>) -> Result<Self, std::io::Error> {
        let dir = tempfile::Builder::new()
            .prefix("harness2-sandbox-")
            .tempdir()?;

        // Validate before writing anything: a rejected tree must leave no files.
        validate_tree_against_root(dir.path(), state)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err.to_string()))?;

        for (path, contents) in state {
            let rel = normalize_rel(path).expect("validated above");
            let destination = dir.path().join(&rel);
            if let Some(parent) = destination.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&destination, contents)?;
        }

        Ok(Self {
            dir,
            executor: None,
            cancel: None,
        })
    }

    /// Test-only constructor with an explicit execution backend, used to prove
    /// fail-closed behaviour without touching the process-wide backend.
    pub fn with_cancel(mut self, cancel: Option<CancellationToken>) -> Self {
        self.cancel = cancel;
        self
    }

    pub fn create_with(
        state: &BTreeMap<PathBuf, String>,
        executor: Arc<dyn Executor>,
    ) -> Result<Self, std::io::Error> {
        let mut workspace = Self::create(state)?;
        workspace.executor = Some(executor);
        Ok(workspace)
    }

    /// The temporary root, for callers that need to inspect or hand off the tree.
    pub fn root(&self) -> &Path {
        self.dir.path()
    }

    /// Resolve a workspace-relative script path, refusing escapes and absolutes.
    fn resolve_script(&self, script_path: &Path) -> Result<PathBuf, SandboxError> {
        if script_path.is_absolute() {
            return Err(SandboxError::InvalidPath {
                path: script_path.to_path_buf(),
                reason: "script path must be relative to the sandbox root",
            });
        }
        let rel = normalize_rel(script_path).ok_or_else(|| SandboxError::InvalidPath {
            path: script_path.to_path_buf(),
            reason: "script path escapes the sandbox root",
        })?;
        let absolute = self.dir.path().join(&rel);
        if !absolute.is_file() {
            return Err(SandboxError::ScriptNotFound {
                path: PathBuf::from(rel),
            });
        }
        Ok(absolute)
    }

    /// Run one script under a wall-clock budget and capture its telemetry.
    ///
    /// A non-zero exit status is a **completed** probe and returns `Ok`: the
    /// caller needs `exit_code` and `stderr` to diagnose it. Infrastructure
    /// failures (missing script, spawn error, timeout) are `Err`. Use
    /// [`Self::run_probe_checked`] when a non-zero status should be an error.
    pub async fn run_probe(
        &self,
        script_path: &PathBuf,
        args: &[&str],
        timeout_ms: u64,
    ) -> Result<ProbeResult, SandboxError> {
        if self
            .cancel
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(SandboxError::Cancelled { partial: Default::default() });
        }
        let executor: Arc<dyn Executor> = match &self.executor {
            Some(executor) => Arc::clone(executor),
            None => Isolation::shared().map_err(map_isolation_error)?,
        };
        let script = self.resolve_script(script_path)?;

        let mut argv: Vec<String> = Vec::with_capacity(args.len() + 1);
        argv.push(script.to_string_lossy().into_owned());
        argv.extend(args.iter().map(|arg| (*arg).to_string()));

        let request = ExecRequest {
            workspace: self.dir.path(),
            program: Path::new(DEFAULT_PYTHON),
            args: &argv,
            timeout_ms,
            cancel: self.cancel.clone(),
        };
        let output = executor.run(&request).await.map_err(map_exec_error)?;

        Ok(ProbeResult {
            exit_code: output.exit_code,
            stdout: output.stdout,
            stderr: output.stderr,
            duration_ms: output.duration_ms,
        })
    }

    /// Like [`Self::run_probe`], but a non-zero exit status becomes
    /// [`SandboxError::NonZeroExit`] carrying the full telemetry.
    pub async fn run_probe_checked(
        &self,
        script_path: &PathBuf,
        args: &[&str],
        timeout_ms: u64,
    ) -> Result<ProbeResult, SandboxError> {
        let result = self.run_probe(script_path, args, timeout_ms).await?;
        if result.exit_code == 0 {
            Ok(result)
        } else {
            Err(SandboxError::NonZeroExit(result))
        }
    }
}

// ---------------------------------------------------------------------------
// Telemetry
// ---------------------------------------------------------------------------

/// One completed probe: what a downstream agent pipeline reads instead of a log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbeResult {
    /// Process exit status; `-1` when terminated by a signal.
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u128,
}

impl ProbeResult {
    pub fn is_success(&self) -> bool {
        self.exit_code == 0
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum SandboxError {
    /// The requested script is not present in the materialised tree.
    ScriptNotFound { path: PathBuf },
    /// The script path is absolute or escapes the workspace.
    InvalidPath { path: PathBuf, reason: &'static str },
    /// The interpreter could not be started.
    Spawn(io::Error),
    /// A pipe or wait operation failed after spawn.
    Io(io::Error),
    /// No isolation backend is usable; candidate code was not executed.
    IsolationUnavailable(String),
    /// The caller cancelled the operation.
    Cancelled { partial: PartialOutput },
    /// A captured stream exceeded its byte budget.
    OutputLimit {
        stream: OutputStream,
        limit_bytes: usize,
        partial: PartialOutput,
    },
    /// Structured execution failure (reader/wait/cleanup) with partial output.
    Execution(Box<ExecError>),
    /// The wall-clock budget expired and the child was killed.
    Timeout {
        timeout_ms: u64,
        duration_ms: u128,
        /// Output captured before the kill, for diagnosing the hang.
        stdout: String,
        stderr: String,
        partial: PartialOutput,
    },
    /// The probe completed with a non-zero status (checked variant only).
    NonZeroExit(ProbeResult),
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ScriptNotFound { path } => {
                write!(f, "probe script not found in sandbox: {}", path.display())
            }
            Self::InvalidPath { path, reason } => {
                write!(f, "invalid probe path {}: {reason}", path.display())
            }
            Self::IsolationUnavailable(reason) => {
                write!(f, "execution isolation unavailable: {reason}")
            }
            Self::Cancelled { .. } => write!(f, "execution was cancelled"),
            Self::OutputLimit {
                stream,
                limit_bytes,
                ..
            } => write!(f, "{stream} exceeded its {limit_bytes}-byte capture limit"),
            Self::Execution(err) => write!(f, "execution failed: {err}"),
            Self::Spawn(err) => write!(f, "could not start isolated process: {err}"),
            Self::Io(err) => write!(f, "probe I/O failure: {err}"),
            Self::Timeout {
                timeout_ms,
                duration_ms,
                ..
            } => write!(
                f,
                "probe exceeded its {timeout_ms} ms budget (ran {duration_ms} ms) and was killed"
            ),
            Self::NonZeroExit(result) => write!(
                f,
                "probe exited with status {}: {}",
                result.exit_code,
                result.stderr.trim()
            ),
        }
    }
}

impl SandboxError {
    pub fn partial(&self) -> Option<&PartialOutput> {
        match self {
            Self::Cancelled { partial }
            | Self::OutputLimit { partial, .. }
            | Self::Timeout { partial, .. } => Some(partial),
            Self::Execution(err) => err.partial(),
            _ => None,
        }
    }

    pub fn cleanup_incomplete(&self) -> bool {
        match self {
            Self::Cancelled { partial }
            | Self::OutputLimit { partial, .. }
            | Self::Timeout { partial, .. } => partial.cleanup_incomplete,
            Self::Execution(err) => err.cleanup_incomplete(),
            _ => false,
        }
    }

    pub fn cleanup_error(&self) -> Option<&str> {
        match self {
            Self::Cancelled { partial }
            | Self::OutputLimit { partial, .. }
            | Self::Timeout { partial, .. } => partial.cleanup_error.as_deref(),
            Self::Execution(err) => err.cleanup_error(),
            _ => None,
        }
    }
}

impl std::error::Error for SandboxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spawn(err) | Self::Io(err) => Some(err),
            Self::Execution(err) => Some(err),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn map_isolation_error(error: ExecError) -> SandboxError {
    match error {
        ExecError::Unavailable(reason) => SandboxError::IsolationUnavailable(reason),
        other => SandboxError::Io(io::Error::other(other.to_string())),
    }
}

fn map_exec_error(error: ExecError) -> SandboxError {
    match error {
        ExecError::Unavailable(reason) => SandboxError::IsolationUnavailable(reason),
        ExecError::Spawn(err) => SandboxError::Spawn(err),
        ExecError::Io(err) => SandboxError::Io(err),
        ExecError::Timeout {
            timeout_ms,
            duration_ms,
            partial,
        } => SandboxError::Timeout {
            timeout_ms,
            duration_ms,
            stdout: partial.stdout.clone(),
            stderr: partial.stderr.clone(),
            partial,
        },
        ExecError::Cancelled { partial } => SandboxError::Cancelled { partial },
        ExecError::OutputLimit {
            stream,
            limit_bytes,
            partial,
        } => SandboxError::OutputLimit {
            stream,
            limit_bytes,
            partial,
        },
        other @ (ExecError::Reader { .. }
        | ExecError::Wait { .. }
        | ExecError::Cleanup { .. }) => SandboxError::Execution(Box::new(other)),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::time::{Duration, Instant};

    use crate::test_executor::HostExecutor;

    fn host_workspace(state: &BTreeMap<PathBuf, String>) -> SandboxWorkspace {
        SandboxWorkspace::create_with(state, Arc::new(HostExecutor)).expect("workspace")
    }

    #[derive(Debug)]
    struct UnavailableExecutor;

    impl Executor for UnavailableExecutor {
        fn run<'a>(
            &'a self,
            _request: &'a ExecRequest<'a>,
        ) -> Pin<Box<dyn Future<Output = Result<crate::execution::ExecOutput, ExecError>> + Send + 'a>>
        {
            Box::pin(async { Err(ExecError::Unavailable("no backend".into())) })
        }
    }

    fn state(entries: &[(&str, &str)]) -> BTreeMap<PathBuf, String> {
        entries
            .iter()
            .map(|(path, contents)| (PathBuf::from(path), (*contents).to_string()))
            .collect()
    }

    const ECHO: &str = r#"
import sys

print("stdout:" + "|".join(sys.argv[1:]))
print("stderr:diagnostic", file=sys.stderr)
sys.exit(3)
"#;

    const SLEEPER: &str = r#"
import pathlib
import time

time.sleep(0.6)
pathlib.Path("late.txt").write_text("finished")
"#;

    #[test]
    fn workspace_materialises_nested_tree_and_cleans_up() {
        let workspace = host_workspace(&state(&[
            ("systemprompt.md", "hello"),
            ("scripts/nested/tool.py", "print('x')\n"),
            ("skills/doc_review/SKILL.md", "---\nname: doc_review\n---\n"),
        ]));

        let root = workspace.root().to_path_buf();
        assert!(root.is_dir());
        assert_eq!(
            std::fs::read_to_string(root.join("scripts/nested/tool.py")).unwrap(),
            "print('x')\n"
        );
        assert!(root.join("skills/doc_review").is_dir());

        drop(workspace);
        assert!(!root.exists(), "temporary tree must be removed on drop");
    }

    #[test]
    fn tree_validation_rejects_duplicates_and_case_aliases() {
        let mut duplicate = BTreeMap::new();
        duplicate.insert(PathBuf::from("scripts/a.py"), "1".to_string());
        duplicate.insert(PathBuf::from("./scripts/a.py"), "2".to_string());
        assert!(matches!(
            validate_tree_paths(&duplicate),
            Err(TreePathError::DuplicatePath { .. })
        ));

        let mut aliased = BTreeMap::new();
        aliased.insert(PathBuf::from("scripts/A.py"), "1".to_string());
        aliased.insert(PathBuf::from("scripts/a.py"), "2".to_string());
        assert!(matches!(
            validate_tree_paths(&aliased),
            Err(TreePathError::CaseAlias { .. })
        ));

        let mut escaping = BTreeMap::new();
        escaping.insert(PathBuf::from("../escape.py"), "1".to_string());
        assert!(matches!(
            validate_tree_paths(&escaping),
            Err(TreePathError::UnsafePath { .. })
        ));
    }

    #[test]
    fn symlinked_destinations_are_rejected() {
        let root = tempfile::tempdir().expect("root");
        let outside = tempfile::tempdir().expect("outside");
        std::os::unix::fs::symlink(outside.path(), root.path().join("link"))
            .expect("symlink");

        let mut state = BTreeMap::new();
        state.insert(PathBuf::from("link/pwn.py"), "print(1)\n".to_string());

        let error = validate_tree_against_root(root.path(), &state).expect_err("symlink");
        assert!(matches!(error, TreePathError::Symlink { .. }));
        assert!(!outside.path().join("pwn.py").exists());
    }

    #[test]
    fn create_rejects_escapes_and_duplicates() {
        let escape = SandboxWorkspace::create(&state(&[("../escape.txt", "nope")]));
        assert!(matches!(escape, Err(err) if err.kind() == io::ErrorKind::InvalidInput));

        let absolute = SandboxWorkspace::create(&state(&[("/tmp/escape.txt", "nope")]));
        assert!(matches!(absolute, Err(err) if err.kind() == io::ErrorKind::InvalidInput));

        let duplicate = SandboxWorkspace::create(&state(&[
            ("scripts/a.py", "one"),
            ("./scripts/a.py", "two"),
        ]));
        assert!(matches!(duplicate, Err(err) if err.kind() == io::ErrorKind::InvalidInput));
    }

    #[test]
    fn exec_error_partial_survives_sandbox_mapping() {
        let partial = PartialOutput {
            stdout: "partial".into(),
            stderr: "err".into(),
            stdout_truncated: true,
            output_incomplete: true,
            cleanup_incomplete: true,
            cleanup_error: Some("reap stuck".into()),
            ..PartialOutput::default()
        };
        let mapped = map_exec_error(ExecError::Wait {
            source: io::Error::other("wait failed"),
            partial: partial.clone(),
        });
        assert!(matches!(mapped, SandboxError::Execution(_)));
        let restored = mapped.partial().expect("partial");
        assert_eq!(restored.stdout, "partial");
        assert!(restored.stdout_truncated);
        assert!(mapped.cleanup_incomplete());
        assert_eq!(mapped.cleanup_error(), Some("reap stuck"));

        let timeout = map_exec_error(ExecError::Timeout {
            timeout_ms: 1,
            duration_ms: 2,
            partial,
        });
        assert!(matches!(timeout, SandboxError::Timeout { .. }));
        assert!(timeout.partial().is_some());
    }

    #[tokio::test]
    async fn unavailable_isolation_fails_closed_without_running_code() {
        let mut state = BTreeMap::new();
        state.insert(PathBuf::from("scripts/never.py"), "print(1)\n".to_string());
        let workspace =
            SandboxWorkspace::create_with(&state, Arc::new(UnavailableExecutor)).expect("workspace");

        let error = workspace
            .run_probe(&PathBuf::from("scripts/never.py"), &[], 1_000)
            .await
            .expect_err("must fail closed");
        assert!(matches!(error, SandboxError::IsolationUnavailable(_)));
    }

    #[tokio::test]
    async fn probe_captures_stdout_stderr_exit_and_duration() {
        let workspace =
            host_workspace(&state(&[("scripts/echo.py", ECHO)]));

        let result = workspace
            .run_probe(&PathBuf::from("scripts/echo.py"), &["alpha", "beta"], 5_000)
            .await
            .expect("probe completes");

        assert_eq!(result.exit_code, 3);
        assert_eq!(result.stdout, "stdout:alpha|beta\n");
        assert_eq!(result.stderr, "stderr:diagnostic\n");
        assert!(!result.is_success());
        // Wall clock is real; only assert it is plausible.
        assert!(result.duration_ms < 5_000);
    }

    #[tokio::test]
    async fn checked_variant_maps_nonzero_exit_to_error() {
        let workspace =
            host_workspace(&state(&[("scripts/echo.py", ECHO)]));

        match workspace
            .run_probe_checked(&PathBuf::from("scripts/echo.py"), &[], 5_000)
            .await
        {
            Err(SandboxError::NonZeroExit(result)) => {
                assert_eq!(result.exit_code, 3);
                assert!(result.stderr.contains("diagnostic"));
            }
            other => panic!("expected NonZeroExit, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn timeout_kills_the_child_and_reports_budget() {
        let workspace =
            host_workspace(&state(&[("scripts/slow.py", SLEEPER)]));
        let root = workspace.root().to_path_buf();

        let started = Instant::now();
        let error = workspace
            .run_probe(&PathBuf::from("scripts/slow.py"), &[], 100)
            .await
            .expect_err("probe must time out");
        let elapsed = started.elapsed();

        match error {
            SandboxError::Timeout {
                timeout_ms,
                duration_ms,
                ..
            } => {
                assert_eq!(timeout_ms, 100);
                assert!(duration_ms < 2_000);
            }
            other => panic!("expected Timeout, got {other:?}"),
        }
        assert!(elapsed < Duration::from_secs(2), "timeout must be prompt");

        // The script would create late.txt after 600 ms. If the kill failed, the
        // file appears; give it well past that deadline before asserting.
        tokio::time::sleep(Duration::from_millis(900)).await;
        assert!(
            !root.join("late.txt").exists(),
            "killed probe must not keep running to completion"
        );
    }

    #[tokio::test]
    async fn missing_and_invalid_scripts_are_distinct_errors() {
        let workspace = host_workspace(&state(&[("scripts/ok.py", "print(1)\n")]));

        match workspace
            .run_probe(&PathBuf::from("scripts/absent.py"), &[], 1_000)
            .await
        {
            Err(SandboxError::ScriptNotFound { path }) => {
                assert_eq!(path, PathBuf::from("scripts/absent.py"));
            }
            other => panic!("expected ScriptNotFound, got {other:?}"),
        }

        match workspace
            .run_probe(&PathBuf::from("/usr/bin/env"), &[], 1_000)
            .await
        {
            Err(SandboxError::InvalidPath { .. }) => {}
            other => panic!("expected InvalidPath, got {other:?}"),
        }

        match workspace
            .run_probe(&PathBuf::from("../outside.py"), &[], 1_000)
            .await
        {
            Err(SandboxError::InvalidPath { .. }) => {}
            other => panic!("expected InvalidPath, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn workspaces_are_isolated_from_each_other() {
        const READER: &str = r#"
import pathlib

print(pathlib.Path("data.txt").read_text())
"#;

        let first = host_workspace(&state(&[
            ("scripts/read.py", READER),
            ("data.txt", "first-workspace"),
        ]));
        let second = host_workspace(&state(&[
            ("scripts/read.py", READER),
            ("data.txt", "second-workspace"),
        ]));

        let a = first
            .run_probe(&PathBuf::from("scripts/read.py"), &[], 5_000)
            .await
            .expect("first probe");
        let b = second
            .run_probe(&PathBuf::from("scripts/read.py"), &[], 5_000)
            .await
            .expect("second probe");

        assert_eq!(a.stdout.trim(), "first-workspace");
        assert_eq!(b.stdout.trim(), "second-workspace");
        assert_ne!(first.root(), second.root());
    }

    #[tokio::test]
    async fn probe_runs_in_workspace_with_a_clean_environment() {
        // The child's cwd is the sandbox root, PYTHONPATH is removed even when the
        // caller sets it, and byte-code writes are disabled.
        const PROBE: &str = r#"
import os
import pathlib

print(os.getcwd())
print(os.environ.get("PYTHONPATH", "<unset>"))
print(os.environ.get("PYTHONDONTWRITEBYTECODE", "<unset>"))
print(pathlib.Path.cwd() == pathlib.Path(os.getcwd()))
"#;
        let workspace =
            host_workspace(&state(&[("scripts/env.py", PROBE)]));
        let result = workspace
            .run_probe(&PathBuf::from("scripts/env.py"), &[], 5_000)
            .await
            .expect("probe");

        let lines: Vec<&str> = result.stdout.lines().collect();
        assert_eq!(lines.len(), 4, "unexpected output: {:?}", result.stdout);
        assert_eq!(
            std::fs::canonicalize(lines[0]).unwrap(),
            std::fs::canonicalize(workspace.root()).unwrap()
        );
        assert_eq!(lines[1], "<unset>", "PYTHONPATH must not be inherited");
        assert_eq!(lines[2], "1");
        assert_eq!(lines[3], "True");
    }
}
