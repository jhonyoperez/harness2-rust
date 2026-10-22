//! Runtime-owned evaluation.
//!
//! The evaluator is deliberately outside the candidate-editable harness tree:
//!
//! * its code lives in `src/evaluator.rs`, not in a `HarnessState`;
//! * its fixtures and expected results live in `eval/`, are loaded by the
//!   runtime, and are **never copied into the candidate workspace** — only the
//!   case input is written there, under a runtime-chosen name;
//! * [`ProtectionPolicy`] rejects candidate mutations that name the evaluator,
//!   the expected results, the fixtures, or this policy, before validation or
//!   probing happens.
//!
//! # What this is not
//!
//! Evaluating a candidate means running its `scripts/summarize.py` with
//! `python3` inside a [`SandboxWorkspace`]. That workspace is a short-lived
//! temporary directory with a wall-clock budget — **not an OS isolation
//! boundary**. There are no namepaces, `setrlimit`, seccomp filters, or
//! container runtime; the child runs with the caller's privileges and can read
//! the host filesystem. A temporary directory or a path validator is not proof
//! of OS isolation, and this module does not claim otherwise. Production use
//! requires replacing the sandbox with a real isolation boundary before
//! executing candidate-written code.
//!
//! # Configuration and fixture limits
//!
//! Loading is fail-closed: [`Evaluator::load`] errors on a missing or malformed
//! `expected.json` and on an empty case list, and [`ProtectionPolicy::load`]
//! errors on a missing or malformed policy. The caller must stop; there is no
//! "evaluation disabled" fallback.
//!
//! The bundled case is a single-case smoke fixture (signed totals vs
//! positive-only totals), **not proof of general correctness** and not an
//! anti-cheat grader. A candidate that hard-codes `Total: 85.15` passes it.
//! Held-out cases are required before this score is used for adoption.
//!
//! # Promotion gap (next stage)
//!
//! The loop evaluates an accepted edit but still promotes it when the fixture
//! fails. Promotion gating and corrective feedback are not implemented here.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;

use crate::execution::{CancellationToken, Executor, PartialOutput};
use crate::sandbox::{validate_tree_paths, SandboxError, SandboxWorkspace, TreePathError};
use crate::validation::normalize_rel;

/// A raw harness snapshot, the same shape the rest of the pipeline uses.
pub type HarnessTree = BTreeMap<PathBuf, String>;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

pub enum EvalError {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// `expected.json` declared no cases, so a pass would be vacuous.
    NoCases,
    /// A fixture path is absolute, escapes the eval root, or is otherwise unsafe.
    InvalidFixture(String),
    Sandbox(SandboxError),
    /// `protected.json` was present but unusable.
    InvalidPolicy(String),
    /// The candidate tree has an unsafe, duplicate, aliased, or symlinked path.
    TreePath(TreePathError),
    /// The candidate tried to occupy the runtime-owned evaluator input file.
    ReservedInput { path: PathBuf },
    /// The evaluator's own script path is unsafe or outside `scripts/`.
    InvalidScriptPath(String),
    /// A declared fixture input is missing, unreadable, or not a regular file.
    Fixture {
        case: String,
        path: PathBuf,
        source: std::io::Error,
    },
    /// A candidate tree names a runtime-owned, protected path.
    Protected(ProtectedViolation),
}

// `main` returns `Box<dyn Error>`, whose terminator prints `Debug`; delegate to
// `Display` so a fail-closed configuration error is readable, not `Io(...)`.
impl std::fmt::Debug for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "evaluation I/O error: {err}"),
            Self::Json(err) => write!(f, "evaluation configuration is invalid JSON: {err}"),
            Self::NoCases => write!(f, "evaluation suite contains no cases"),
            Self::InvalidFixture(detail) => write!(f, "invalid evaluation fixture: {detail}"),
            Self::Sandbox(err) => write!(f, "candidate execution failed: {err}"),
            Self::InvalidPolicy(detail) => write!(f, "invalid protection policy: {detail}"),
            Self::TreePath(err) => write!(f, "candidate tree rejected: {err}"),
            Self::ReservedInput { path } => write!(
                f,
                "{} is reserved for runtime-owned evaluation input and cannot be supplied by a candidate",
                path.display()
            ),
            Self::InvalidScriptPath(detail) => write!(f, "invalid evaluator script path: {detail}"),
            Self::Fixture { case, path, source } => write!(
                f,
                "fixture input for case {case:?} at {} is unusable: {source}",
                path.display()
            ),
            Self::Protected(violation) => write!(f, "candidate rejected: {violation}"),
        }
    }
}

impl Error for EvalError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Json(err) => Some(err),
            Self::Sandbox(err) => Some(err),
            Self::TreePath(err) => Some(err),
            Self::Fixture { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl EvalError {
    pub fn partial(&self) -> Option<&PartialOutput> {
        match self {
            Self::Sandbox(error) => error.partial(),
            _ => None,
        }
    }

    pub fn cleanup_incomplete(&self) -> bool {
        match self {
            Self::Sandbox(error) => error.cleanup_incomplete(),
            _ => false,
        }
    }
}

impl From<std::io::Error> for EvalError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<serde_json::Error> for EvalError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

impl From<TreePathError> for EvalError {
    fn from(err: TreePathError) -> Self {
        Self::TreePath(err)
    }
}

impl From<ProtectedViolation> for EvalError {
    fn from(err: ProtectedViolation) -> Self {
        Self::Protected(err)
    }
}

// ---------------------------------------------------------------------------
// Fixtures and results
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ExpectedCase {
    pub name: String,
    /// Path relative to the eval root, e.g. `cases/ledger.csv`.
    pub input: String,
    pub expected_stdout: String,
}

#[derive(Debug, Deserialize)]
struct ExpectedSuite {
    cases: Vec<ExpectedCase>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaseOutcome {
    pub name: String,
    pub passed: bool,
    pub expected: String,
    pub actual: String,
    pub exit_code: i32,
    pub stderr: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalReport {
    pub outcomes: Vec<CaseOutcome>,
}

impl EvalReport {
    pub fn passed(&self) -> bool {
        !self.outcomes.is_empty() && self.outcomes.iter().all(|case| case.passed)
    }

    pub fn failures(&self) -> usize {
        self.outcomes.iter().filter(|case| !case.passed).count()
    }

    /// Failure detail that is safe to return to the model: case names, exit
    /// codes, and the candidate's own output — never `expected_stdout`.
    pub fn public_summary(&self) -> String {
        if self.passed() {
            return format!("{}/{} cases passed", self.outcomes.len(), self.outcomes.len());
        }
        let details = self
            .outcomes
            .iter()
            .filter(|case| !case.passed)
            .map(|case| {
                let actual: String = case.actual.chars().take(120).collect();
                format!("{}: exit {} output {:?}", case.name, case.exit_code, actual)
            })
            .collect::<Vec<_>>()
            .join("; ");
        format!(
            "{}/{} cases passed; {details}",
            self.outcomes.len() - self.failures(),
            self.outcomes.len()
        )
    }

    pub fn summary(&self) -> String {
        if self.passed() {
            format!("{}/{} cases passed", self.outcomes.len(), self.outcomes.len())
        } else {
            let detail = self
                .outcomes
                .iter()
                .filter(|case| !case.passed)
                .map(|case| {
                    format!(
                        "{}: expected {:?}, got {:?} (exit {})",
                        case.name, case.expected, case.actual, case.exit_code
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            format!(
                "{}/{} cases passed; {detail}",
                self.outcomes.len() - self.failures(),
                self.outcomes.len()
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

/// A fixture input read once during initialisation. Evaluation never touches
/// the fixture tree again, so a later deletion or permission change cannot turn
/// a configuration error into a post-model execution error.
#[derive(Debug, Clone)]
struct LoadedCase {
    case: ExpectedCase,
    input: Vec<u8>,
}

/// Loads protected cases once, then runs a candidate tree against each of them.
#[derive(Debug, Clone)]
pub struct Evaluator {
    root: PathBuf,
    script: PathBuf,
    timeout_ms: u64,
    policy: ProtectionPolicy,
    cases: Vec<LoadedCase>,
    executor: Option<Arc<dyn Executor>>,
    cancel: Option<CancellationToken>,
}

impl Evaluator {
    /// Default candidate script under test.
    pub const DEFAULT_SCRIPT: &'static str = "scripts/summarize.py";

    /// Load `expected.json`, read every fixture input, and use the mandatory
    /// protection policy.
    pub fn load(root: impl AsRef<Path>) -> Result<Self, EvalError> {
        Self::load_with_policy(
            root,
            PathBuf::from(Self::DEFAULT_SCRIPT),
            30_000,
            ProtectionPolicy::builtin(),
        )
    }

    /// Stable digest of the runtime-owned evaluation configuration. It binds
    /// checkpoint evidence to this exact script, fixture set, limits, and
    /// protection policy without serializing expected answers.
    pub fn configuration_digest(&self) -> String {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"harness2-evaluator-v1");
        bytes.extend_from_slice(self.root.to_string_lossy().as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(self.script.to_string_lossy().as_bytes());
        bytes.extend_from_slice(&self.timeout_ms.to_le_bytes());
        for prefix in &self.policy.prefixes {
            bytes.extend_from_slice(prefix.as_bytes());
            bytes.push(0);
        }
        for case in &self.cases {
            bytes.extend_from_slice(case.case.name.as_bytes());
            bytes.push(0);
            bytes.extend_from_slice(case.case.input.as_bytes());
            bytes.push(0);
            bytes.extend_from_slice(case.case.expected_stdout.as_bytes());
            bytes.push(0);
            bytes.extend_from_slice(&case.input);
            bytes.push(0);
        }
        crate::checkpoint::sha256_hex(&bytes)
    }

    pub fn load_with(
        root: impl AsRef<Path>,
        script: PathBuf,
        timeout_ms: u64,
    ) -> Result<Self, EvalError> {
        Self::load_with_policy(root, script, timeout_ms, ProtectionPolicy::builtin())
    }

    /// Full constructor: configured policy, script, timeout, and eagerly loaded
    /// fixture inputs. A missing, unreadable, or non-file input fails here,
    /// before any candidate tree or model session exists.
    pub fn load_with_policy(
        root: impl AsRef<Path>,
        script: PathBuf,
        timeout_ms: u64,
        policy: ProtectionPolicy,
    ) -> Result<Self, EvalError> {
        let root = root.as_ref().to_path_buf();
        let script = validated_script(&script)?;
        let raw = std::fs::read_to_string(root.join("expected.json"))?;
        let suite: ExpectedSuite = serde_json::from_str(&raw)?;
        if suite.cases.is_empty() {
            return Err(EvalError::NoCases);
        }

        let mut cases = Vec::with_capacity(suite.cases.len());
        for case in suite.cases {
            let relative = Path::new(&case.input);
            let safe = !relative.is_absolute()
                && relative
                    .components()
                    .all(|part| matches!(part, Component::Normal(_) | Component::CurDir));
            if !safe {
                return Err(EvalError::InvalidFixture(format!(
                    "case {:?} input {:?} must stay under the eval root",
                    case.name, case.input
                )));
            }

            let path = root.join(&case.input);
            let metadata = std::fs::metadata(&path).map_err(|source| EvalError::Fixture {
                case: case.name.clone(),
                path: path.clone(),
                source,
            })?;
            if !metadata.is_file() {
                return Err(EvalError::Fixture {
                    case: case.name.clone(),
                    path: path.clone(),
                    source: std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "fixture input is not a regular file",
                    ),
                });
            }
            let input = std::fs::read(&path).map_err(|source| EvalError::Fixture {
                case: case.name.clone(),
                path: path.clone(),
                source,
            })?;
            cases.push(LoadedCase { case, input });
        }

        Ok(Self {
            root,
            script,
            timeout_ms,
            policy,
            cases,
            executor: None,
            cancel: None,
        })
    }

    pub fn cases(&self) -> impl Iterator<Item = &ExpectedCase> + '_ {
        self.cases.iter().map(|loaded| &loaded.case)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Pure validation of a candidate tree: no filesystem access, no execution.
    ///
    /// Every materialisation path calls this (or the identical sandbox check)
    /// before creating a workspace, so a rejected tree cannot run a script.
    pub fn with_cancel(mut self, cancel: Option<CancellationToken>) -> Self {
        self.cancel = cancel;
        self
    }

    /// Inject an executor (tests only; production uses `Isolation::shared`).
    pub fn with_executor(mut self, executor: Arc<dyn Executor>) -> Self {
        self.executor = Some(executor);
        self
    }

    pub fn validate_candidate(&self, tree: &HarnessTree) -> Result<(), EvalError> {
        validate_tree_paths(tree)?;

        // Direct `evaluate` callers bypass `run_loop`'s policy gate, so the
        // mandatory (and configured) protections are re-applied here.
        self.policy
            .check_paths(tree.keys())
            .map_err(EvalError::Protected)?;

        for path in tree.keys() {
            // Normalise exactly as materialisation does before comparing the
            // reserved name; `file_name()` on the raw key would miss aliases
            // such as `.\eval_input.csv`.
            let Some(relative) = normalize_rel(path) else {
                continue;
            };
            let reserved = Path::new(&relative)
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.eq_ignore_ascii_case("eval_input.csv"));
            if reserved {
                return Err(EvalError::ReservedInput { path: path.clone() });
            }
        }
        Ok(())
    }

    /// Run the candidate's script once per protected case and compare exact
    /// stdout. The expected values never enter the candidate workspace.
    pub async fn evaluate(&self, tree: &HarnessTree) -> Result<EvalReport, EvalError> {
        self.evaluate_with_cancel(tree, self.cancel.as_ref()).await
    }

    /// Evaluate with one explicit cancellation context shared with the caller.
    /// A pre-cancelled context returns before any workspace or launch exists.
    pub async fn evaluate_with_cancel(
        &self,
        tree: &HarnessTree,
        cancel: Option<&CancellationToken>,
    ) -> Result<EvalReport, EvalError> {
        if cancel.is_some_and(CancellationToken::is_cancelled) {
            return Err(EvalError::Sandbox(SandboxError::Cancelled { partial: Default::default() }));
        }
        // Fail closed before any workspace exists or any script can run.
        self.validate_candidate(tree)?;

        let mut outcomes = Vec::with_capacity(self.cases.len());

        for loaded in &self.cases {
            if cancel.is_some_and(CancellationToken::is_cancelled) {
                return Err(EvalError::Sandbox(SandboxError::Cancelled { partial: Default::default() }));
            }
            let case = &loaded.case;
            let workspace = match &self.executor {
                Some(executor) => SandboxWorkspace::create_with(tree, Arc::clone(executor))?,
                None => SandboxWorkspace::create(tree)?,
            };
            let workspace = workspace.with_cancel(cancel.cloned());

            // Runtime-owned input: written after the candidate tree, under a
            // name the candidate cannot influence, so an existing file with the
            // same name is overwritten rather than trusted.
            let input_name = "eval_input.csv";
            std::fs::write(workspace.root().join(input_name), &loaded.input)?;

            let probe = workspace
                .run_probe(&self.script, &[input_name], self.timeout_ms)
                .await
                .map_err(EvalError::Sandbox)?;

            let passed = probe.exit_code == 0 && probe.stdout == case.expected_stdout;
            outcomes.push(CaseOutcome {
                name: case.name.clone(),
                passed,
                expected: case.expected_stdout.clone(),
                actual: probe.stdout,
                exit_code: probe.exit_code,
                stderr: probe.stderr,
            });
        }

        Ok(EvalReport { outcomes })
    }
}

// ---------------------------------------------------------------------------
// Protection policy
// ---------------------------------------------------------------------------

/// Paths a candidate may never write, delete, or rename.
///
/// Prefixes are repository-relative. Matching is case-insensitive because the
/// default macOS filesystem is, and normalisation rejects absolute paths and
/// `..` escapes — an escaping path is a violation, never "unprotected".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtectionPolicy {
    prefixes: Vec<String>,
}

impl Default for ProtectionPolicy {
    fn default() -> Self {
        Self::builtin()
    }
}

impl ProtectionPolicy {
    /// Prefixes that cannot be removed by configuration.
    pub const MANDATORY: [&'static str; 6] =
        ["eval", "src", "tests", "cargo.toml", "cargo.lock", ".git"];

    pub fn builtin() -> Self {
        Self {
            prefixes: Self::MANDATORY.iter().map(|p| (*p).to_string()).collect(),
        }
    }

    /// Load `{"protected": ["eval", "src", ...]}`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, EvalError> {
        #[derive(Deserialize)]
        struct Document {
            protected: Vec<String>,
        }

        let raw = std::fs::read_to_string(path.as_ref())?;
        let document: Document = serde_json::from_str(&raw)?;

        // Configuration may ADD protections; the mandatory set is always unioned
        // back in, so a hostile or partial file cannot narrow the policy.
        let mut prefixes: Vec<String> = Self::MANDATORY.iter().map(|p| (*p).to_string()).collect();
        for entry in document.protected {
            let normalized = normalize_rel(Path::new(&entry)).ok_or_else(|| {
                EvalError::InvalidPolicy(format!("protected entry {entry:?} is not relative"))
            })?;
            prefixes.push(normalized.to_ascii_lowercase());
        }
        prefixes.sort();
        prefixes.dedup();
        Ok(Self { prefixes })
    }

    pub fn prefixes(&self) -> &[String] {
        &self.prefixes
    }

    /// Reject the first path that is protected or that escapes the tree.
    pub fn check_paths<'a, I>(&self, paths: I) -> Result<(), ProtectedViolation>
    where
        I: IntoIterator<Item = &'a PathBuf>,
    {
        for path in paths {
            let Some(relative) = normalize_rel(path) else {
                return Err(ProtectedViolation {
                    path: path.clone(),
                    matched: "<escapes the tree>".to_string(),
                });
            };
            let lowered = relative.to_ascii_lowercase();
            for prefix in &self.prefixes {
                if lowered == *prefix || lowered.starts_with(&format!("{prefix}/")) {
                    return Err(ProtectedViolation {
                        path: path.clone(),
                        matched: prefix.clone(),
                    });
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtectedViolation {
    pub path: PathBuf,
    pub matched: String,
}

impl std::fmt::Display for ProtectedViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} is runtime-owned (matched {:?}) and cannot be edited by a candidate",
            self.path.display(),
            self.matched
        )
    }
}

impl Error for ProtectedViolation {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The evaluator's own script must be a relative `.py` under `scripts/`.
fn validated_script(script: &Path) -> Result<PathBuf, EvalError> {
    if script.is_absolute() {
        return Err(EvalError::InvalidScriptPath(format!(
            "{} is absolute",
            script.display()
        )));
    }
    let normalized = normalize_rel(script).ok_or_else(|| {
        EvalError::InvalidScriptPath(format!("{} escapes the harness root", script.display()))
    })?;
    let lowered = normalized.to_ascii_lowercase();
    if !lowered.starts_with("scripts/") || !lowered.ends_with(".py") {
        return Err(EvalError::InvalidScriptPath(format!(
            "{} must be a relative .py file under scripts/",
            script.display()
        )));
    }
    Ok(PathBuf::from(normalized))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_executor::HostExecutor;

    const EVAL_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/eval");

    /// The corrected implementation: sums signed amounts.
    const CORRECT: &str = r#"import csv
import io
import sys
from decimal import Decimal

DEFAULT_LEDGER = "amount\n100.25\n-20.10\n5.00\n"


def total(text):
    rows = csv.DictReader(io.StringIO(text))
    return sum((Decimal(row["amount"]) for row in rows), Decimal(0))


if len(sys.argv) > 1:
    with open(sys.argv[1]) as source:
        text = source.read()
else:
    text = DEFAULT_LEDGER

print(f"Total: {total(text):.2f}")
"#;

    /// The deliberately wrong implementation: drops negative amounts.
    const BROKEN: &str = r#"import csv
import io
import sys
from decimal import Decimal

DEFAULT_LEDGER = "amount\n100.25\n-20.10\n5.00\n"


def total(text):
    rows = csv.DictReader(io.StringIO(text))
    return sum((Decimal(row["amount"]) for row in rows if Decimal(row["amount"]) > 0), Decimal(0))


if len(sys.argv) > 1:
    with open(sys.argv[1]) as source:
        text = source.read()
else:
    text = DEFAULT_LEDGER

print(f"Total: {total(text):.2f}")
"#;

    fn tree(script: &str) -> HarnessTree {
        let mut files = BTreeMap::new();
        files.insert(PathBuf::from("scripts/summarize.py"), script.to_string());
        files
    }

    fn evaluator() -> Evaluator {
        Evaluator::load(EVAL_ROOT)
            .expect("evaluator loads")
            .with_executor(Arc::new(HostExecutor))
    }

    fn tree_from(entries: &[(&str, &str)]) -> HarnessTree {
        entries
            .iter()
            .map(|(path, contents)| (PathBuf::from(path), (*contents).to_string()))
            .collect()
    }

    fn policy() -> ProtectionPolicy {
        ProtectionPolicy::load(Path::new(EVAL_ROOT).join("protected.json")).expect("policy loads")
    }

    #[tokio::test]
    async fn fixture_inputs_are_loaded_during_initialization() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("cases")).expect("cases dir");
        std::fs::write(
            dir.path().join("expected.json"),
            r#"{"cases":[{"name":"c","input":"cases/in.csv","expected_stdout":"Total: 85.15\n"}]}"#,
        )
        .expect("expected");

        // Missing input is a configuration error at load time.
        let error = Evaluator::load(dir.path()).expect_err("missing fixture input");
        assert!(matches!(error, EvalError::Fixture { .. }));

        // A directory is not a readable fixture file.
        std::fs::create_dir(dir.path().join("cases/in.csv")).expect("decoy dir");
        let error = Evaluator::load(dir.path()).expect_err("directory fixture input");
        assert!(matches!(error, EvalError::Fixture { .. }));
        std::fs::remove_dir(dir.path().join("cases/in.csv")).expect("remove decoy");

        // Valid input, deleted after load: evaluation must use the bytes read
        // during initialisation and never touch the fixture tree again.
        std::fs::write(
            dir.path().join("cases/in.csv"),
            "amount\n100.25\n-20.10\n5.00\n",
        )
        .expect("input");
        let evaluator = Evaluator::load(dir.path())
            .expect("evaluator loads")
            .with_executor(Arc::new(HostExecutor));
        std::fs::remove_file(dir.path().join("cases/in.csv")).expect("delete after load");

        let report = evaluator.evaluate(&tree(CORRECT)).await.expect("evaluate");
        assert!(report.passed(), "{}", report.summary());
    }

    #[tokio::test]
    async fn reserved_input_backslash_alias_is_rejected_before_execution() {
        let evaluator = evaluator();
        let marker = std::env::temp_dir().join(format!(
            "harness2-eval-alias-marker-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&marker);
        let script = format!(
            "import pathlib\npathlib.Path({:?}).write_text('ran')\n",
            marker.to_string_lossy()
        );

        for alias in [".\\eval_input.csv", "./eval_input.csv"] {
            let candidate = tree_from(&[
                ("scripts/summarize.py", script.as_str()),
                (alias, ""),
            ]);
            let error = evaluator
                .evaluate(&candidate)
                .await
                .expect_err("reserved alias");
            assert!(
                matches!(error, EvalError::ReservedInput { .. }),
                "{alias}: {error}"
            );
        }
        assert!(
            !marker.exists(),
            "reserved-name validation must run before the script"
        );
        let _ = std::fs::remove_file(&marker);
    }

    #[tokio::test]
    async fn direct_evaluate_rejects_mandatory_protected_paths() {
        let evaluator = evaluator();
        for protected in [
            "eval/expected.json",
            "src/main.rs",
            "Cargo.toml",
            "tests/protected.rs",
            ".git/config",
        ] {
            let candidate =
                tree_from(&[("scripts/summarize.py", CORRECT), (protected, "x")]);

            assert!(
                matches!(
                    evaluator.validate_candidate(&candidate),
                    Err(EvalError::Protected(_))
                ),
                "{protected} must fail validation"
            );
            let error = evaluator
                .evaluate(&candidate)
                .await
                .expect_err("protected path");
            assert!(
                matches!(error, EvalError::Protected(_)),
                "{protected}: {error}"
            );
        }
    }

    #[test]
    fn mandatory_protections_cannot_be_removed_by_configuration() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("protected.json");
        // A configuration that tries to protect a single unrelated path.
        std::fs::write(&path, r#"{"protected":["notes"]}"#).expect("policy file");

        let policy = ProtectionPolicy::load(&path).expect("policy loads");
        let rejected = |candidate: &str| {
            let candidate = PathBuf::from(candidate);
            policy.check_paths(std::iter::once(&candidate)).is_err()
        };

        for required in ProtectionPolicy::MANDATORY {
            assert!(rejected(required), "{required} must stay protected");
            assert!(
                rejected(&format!("{required}/nested")),
                "{required}/nested must stay protected"
            );
        }
        assert!(rejected("notes/readme.md"));
        assert!(!rejected("systemprompt.md"));
    }

    #[test]
    fn evaluator_load_fails_closed_on_missing_malformed_or_empty_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(matches!(Evaluator::load(dir.path()), Err(EvalError::Io(_))));

        std::fs::write(dir.path().join("expected.json"), "{").unwrap();
        assert!(matches!(
            Evaluator::load(dir.path()),
            Err(EvalError::Json(_))
        ));

        std::fs::write(dir.path().join("expected.json"), r#"{"cases":[]}"#).unwrap();
        assert!(matches!(
            Evaluator::load(dir.path()),
            Err(EvalError::NoCases)
        ));
    }

    #[test]
    fn evaluator_script_paths_are_validated() {
        for bad in ["../x.py", "/tmp/x.py", "scripts/x.txt", "x.py", ""] {
            let result = Evaluator::load_with(EVAL_ROOT, PathBuf::from(bad), 1_000);
            assert!(
                matches!(result, Err(EvalError::InvalidScriptPath(_))),
                "{bad:?} must be rejected"
            );
        }
        assert!(Evaluator::load_with(
            EVAL_ROOT,
            PathBuf::from("scripts/summarize.py"),
            1_000
        )
        .is_ok());
    }

    #[test]
    fn candidate_validation_rejects_reserved_duplicate_case_and_escape_paths() {
        let evaluator = evaluator();

        assert!(matches!(
            evaluator.validate_candidate(&tree_from(&[("eval_input.csv", "")])),
            Err(EvalError::ReservedInput { .. })
        ));
        assert!(matches!(
            evaluator.validate_candidate(&tree_from(&[
                ("scripts/a.py", "1"),
                ("./scripts/a.py", "2")
            ])),
            Err(EvalError::TreePath(TreePathError::DuplicatePath { .. }))
        ));
        assert!(matches!(
            evaluator.validate_candidate(&tree_from(&[
                ("scripts/A.py", "1"),
                ("scripts/a.py", "2")
            ])),
            Err(EvalError::TreePath(TreePathError::CaseAlias { .. }))
        ));
        assert!(matches!(
            evaluator.validate_candidate(&tree_from(&[("../escape.py", "1")])),
            Err(EvalError::TreePath(TreePathError::UnsafePath { .. }))
        ));
    }

    #[tokio::test]
    async fn direct_evaluate_rejects_before_any_script_runs() {
        let evaluator = evaluator();
        let marker = std::env::temp_dir().join(format!(
            "harness2-eval-marker-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&marker);

        let script = format!(
            "import pathlib\npathlib.Path({:?}).write_text('ran')\n",
            marker.to_string_lossy()
        );
        let candidate = tree_from(&[
            ("scripts/summarize.py", script.as_str()),
            ("eval_input.csv", ""),
        ]);

        let error = evaluator
            .evaluate(&candidate)
            .await
            .expect_err("reserved input must be rejected");
        assert!(matches!(error, EvalError::ReservedInput { .. }));
        assert!(
            !marker.exists(),
            "validation must happen before the script can run"
        );
        let _ = std::fs::remove_file(&marker);
    }

    #[tokio::test]
    async fn corrected_candidate_passes() {
        let report = evaluator().evaluate(&tree(CORRECT)).await.expect("eval runs");
        assert!(report.passed(), "{}", report.summary());
        assert_eq!(report.outcomes.len(), 1);
        assert_eq!(report.outcomes[0].actual, "Total: 85.15\n");
        assert_eq!(report.failures(), 0);
    }

    #[tokio::test]
    async fn deliberately_wrong_candidate_fails() {
        let report = evaluator().evaluate(&tree(BROKEN)).await.expect("eval runs");
        assert!(!report.passed(), "positive-only totals must not pass");
        assert_eq!(report.failures(), 1);
        assert_eq!(report.outcomes[0].actual, "Total: 105.25\n");
        assert_ne!(
            report.outcomes[0].actual,
            report.outcomes[0].expected
        );
        assert!(report.summary().contains("signed-ledger-total"));
    }

    #[tokio::test]
    async fn a_missing_implementation_is_an_execution_error() {
        let error = evaluator()
            .evaluate(&BTreeMap::new())
            .await
            .expect_err("no script");
        assert!(matches!(error, EvalError::Sandbox(_)));
    }

    #[test]
    fn protection_rejects_evaluator_and_escape_paths() {
        let policy = policy();
        for protected in [
            "eval/expected.json",
            "eval/cases/ledger.csv",
            "eval/protected.json",
            "src/evaluator.rs",
            "Cargo.toml",
            ".git/config",
            "../src/main.rs",
            "/etc/passwd",
        ] {
            let path = PathBuf::from(protected);
            assert!(
                policy.check_paths(std::iter::once(&path)).is_err(),
                "{protected} must be rejected"
            );
        }

        for allowed in [
            "systemprompt.md",
            "scripts/summarize.py",
            "evalish/notes.md",
            "srcish/x.py",
        ] {
            let path = PathBuf::from(allowed);
            assert!(
                policy.check_paths(std::iter::once(&path)).is_ok(),
                "{allowed} should be editable"
            );
        }
    }

    #[test]
    fn policy_prefixes_are_loaded_from_the_protected_file() {
        let policy = policy();
        assert!(policy.prefixes().contains(&"eval".to_string()));
        assert!(policy.prefixes().contains(&"src".to_string()));
        assert!(policy.prefixes().contains(&"cargo.toml".to_string()));
    }

    /// Containment for the evaluator path. Ignored because the development
    /// environment cannot nest `sandbox-exec`; run on an unsandboxed host with
    /// `cargo test --ignored`. Deliberately does **not** inject an executor:
    /// it must use production `Isolation::shared()`, never a host test helper.
    /// Unavailable containment fails; it is never reported as passed.
    #[tokio::test]
    #[ignore = "requires an unsandboxed host; sandbox-exec cannot be nested"]
    async fn evaluator_containment_checks_run_inside_isolation() {
        let canary_dir = tempfile::tempdir().expect("canaries");
        let canary_read = canary_dir.path().join("read-canary.txt");
        let canary_write = canary_dir.path().join("write-canary.txt");
        std::fs::write(&canary_read, "canary").expect("read canary");
        std::fs::write(&canary_write, "intact").expect("write canary");

        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("listener");
        let port = listener.local_addr().expect("addr").port();
        // Control connection from outside the candidate sandbox: proves the
        // endpoint is reachable before a denial is interpreted as policy.
        std::net::TcpStream::connect(("127.0.0.1", port))
            .expect("control connection")
            .shutdown(std::net::Shutdown::Both)
            .expect("control shutdown");

        let script = format!(
            r#"import os, pathlib, socket
def denied(fn):
    try:
        fn(); return False
    except PermissionError:
        return True
    except Exception:
        return False
read_canary, write_canary = pathlib.Path("canaries.txt").read_text().splitlines()
host = denied(lambda: pathlib.Path(read_canary).read_text())
outside = denied(lambda: pathlib.Path(write_canary).write_text("tampered"))
sentinel = os.environ.get("HARNESS2_SENTINEL_EVAL") is None
port = int(pathlib.Path("net_port.txt").read_text())
try:
    sock = socket.create_connection(("127.0.0.1", port), timeout=1)
    sock.close()
    network = "connected"
except PermissionError:
    network = "policy"
except OSError as exc:
    network = f"other:{{exc.errno}}"
print("Total: 85.15" if host and outside and sentinel and network == "policy" else "ESCAPED")
"#
        );

        let mut tree = BTreeMap::new();
        tree.insert(PathBuf::from("scripts/summarize.py"), script);
        tree.insert(
            PathBuf::from("canaries.txt"),
            format!(
                "{}\n{}\n",
                canary_read.display(),
                canary_write.display()
            ),
        );
        tree.insert(PathBuf::from("net_port.txt"), port.to_string());

        std::env::set_var("HARNESS2_SENTINEL_EVAL", "leaked");
        let evaluator = Evaluator::load(EVAL_ROOT).expect("evaluator loads");
        let report = evaluator
            .evaluate(&tree)
            .await
            .expect("contained evaluation");
        std::env::remove_var("HARNESS2_SENTINEL_EVAL");
        assert!(report.passed(), "{}", report.summary());
        assert_eq!(
            std::fs::read_to_string(&canary_write).expect("canary"),
            "intact"
        );
    }

    #[derive(Debug)]
    struct UnavailableExecutor;

    impl Executor for UnavailableExecutor {
        fn run<'a>(
            &'a self,
            _request: &'a crate::execution::ExecRequest<'a>,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<crate::execution::ExecOutput, crate::execution::ExecError>,
                    > + Send
                    + 'a,
            >,
        > {
            Box::pin(async {
                Err(crate::execution::ExecError::Unavailable(
                    "no backend".into(),
                ))
            })
        }
    }

    #[test]
    fn eval_error_preserves_sandbox_partial_and_cleanup_state() {
        let partial = PartialOutput {
            stdout: "partial".into(),
            output_incomplete: true,
            cleanup_incomplete: true,
            cleanup_error: Some("reap stuck".into()),
            ..PartialOutput::default()
        };
        let error = EvalError::Sandbox(SandboxError::Timeout {
            timeout_ms: 1,
            duration_ms: 2,
            stdout: partial.stdout.clone(),
            stderr: String::new(),
            partial,
        });
        assert_eq!(error.partial().expect("partial").stdout, "partial");
        assert!(error.cleanup_incomplete());
    }

    #[tokio::test]
    async fn unavailable_executor_fails_evaluation_closed() {
        let evaluator = Evaluator::load(EVAL_ROOT)
            .expect("evaluator loads")
            .with_executor(Arc::new(UnavailableExecutor));
        let error = evaluator
            .evaluate(&tree(CORRECT))
            .await
            .expect_err("isolation failure must not pass");
        assert!(matches!(
            error,
            EvalError::Sandbox(SandboxError::IsolationUnavailable(_))
        ));
    }

    #[tokio::test]
    async fn protected_files_stay_byte_identical_across_evaluation() {
        let before = snapshot(Path::new(EVAL_ROOT));
        assert!(!before.is_empty(), "fixture snapshot must not be empty");

        let _ = evaluator().evaluate(&tree(CORRECT)).await.expect("eval runs");
        let _ = evaluator().evaluate(&tree(BROKEN)).await.expect("eval runs");

        let after = snapshot(Path::new(EVAL_ROOT));
        assert_eq!(before, after, "evaluation must not mutate eval/");
    }

    fn snapshot(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
        let mut out = BTreeMap::new();
        walk(dir, dir, &mut out);
        out
    }

    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<PathBuf, Vec<u8>>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(root, &path, out);
            } else if let Ok(bytes) = std::fs::read(&path) {
                out.insert(path.strip_prefix(root).unwrap_or(&path).to_path_buf(), bytes);
            }
        }
    }
}
