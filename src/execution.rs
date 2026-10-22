//! One execution interface for every candidate-code path.
//!
//! Both the preliminary probe (`SandboxWorkspace::run_probe`) and the evaluator
//! (`Evaluator::evaluate`) ultimately call [`Executor::run`]. A normal library
//! build contains exactly one executor — macOS `sandbox-exec` — and no host
//! fallback. If detection fails, execution fails closed and no candidate code
//! runs.
//!
//! # Supervision model
//!
//! The fixed launcher is called synchronously and cannot be preempted; its time
//! counts against the deadline. Once spawned, a [`Supervision`] guard owns the
//! process group, the reader tasks, and cleanup. Dropping or aborting the
//! execution future signals the group, starts the leader kill, aborts both
//! readers, and spawns a reaper so cleanup does not depend on the caller
//! awaiting anything. Cleanup is guarded and idempotent.
//!
//! One absolute deadline is checked before launch, during wait/drain, at the
//! success boundary, and before completion; cleanup has its own bounded grace
//! and reports `cleanup_incomplete` if that grace expires. stdout/stderr are captured through
//! bounded shared buffers (1 MiB each); timeout, cancellation, overflow, and
//! reader errors return capped partial output with `output_incomplete` and
//! per-stream truncation flags. Cleanup has its own bounded wait; if it does not
//! finish, the error says `cleanup_incomplete` and a background reaper retains
//! responsibility for the group.
//!
//! # Backend capabilities (sandbox-exec / Seatbelt)
//!
//! The profile denies by default and then allows only:
//!
//! * `process-exec` for the Python shim and the Command Line Tools interpreter
//!   it resolves to;
//! * reads of `/usr/lib`, `/usr/share/locale`, `/System/Library/dyld`,
//!   `/System/Library/Frameworks`, the Python framework under
//!   `/Library/Developer/CommandLineTools`, `/private/var/db/timezone`,
//!   `/dev/null`, `/dev/urandom`, and the run workspace;
//! * writes under the run workspace (and `/dev/null`);
//! * **no network** (`(deny network*)`, on top of the default deny).
//!
//! Deliberately absent: global metadata reads, `process-fork`, `signal`,
//! `sysctl-read`, home directories, repository paths, sockets, and expected
//! answers. The child starts with a cleared environment, as the current
//! non-root user, with `RLIMIT_CPU`, `RLIMIT_FSIZE`, and `RLIMIT_NOFILE`.
//!
//! # Verification status
//!
//! `containment_verified` is `false` until the containment tests run on an
//! unsandboxed host. This environment cannot nest `sandbox-exec`, so the real
//! containment and real supervision paths are unverified; the supervision
//! tests below are process mocks.

use std::future::Future;
use std::io::{self, Read};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};
use tokio::sync::Notify;
use tokio::task::{JoinError, JoinHandle};
use tokio::time::{sleep, timeout};

/// Absolute macOS sandbox launcher.
pub const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";
/// System Python shim used inside the sandbox.
pub const DEFAULT_PYTHON: &str = "/usr/bin/python3";
/// Interpreter the shim resolves to on this host; execution is restricted to it.
pub const CLT_PYTHON: &str = "/Library/Developer/CommandLineTools/usr/bin/python3";
/// Framework binary the CLT shim links to.
pub const FRAMEWORK_PYTHON: &str =
    "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3";

const MAX_STDOUT_FILESIZE: u64 = 8 * 1024 * 1024;
const MAX_OPEN_FILES: u64 = 64;
const MAX_CAPTURE_BYTES: usize = 1024 * 1024;
const CLEANUP_GRACE: Duration = Duration::from_secs(1);
const SELF_TEST_TIMEOUT: Duration = Duration::from_secs(3);
const POLL_INTERVAL: Duration = Duration::from_millis(2);

/// One execution request. `workspace` is the only writable tree exposed.
pub struct ExecRequest<'a> {
    pub workspace: &'a Path,
    pub program: &'a Path,
    pub args: &'a [String],
    pub timeout_ms: u64,
    pub cancel: Option<CancellationToken>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecOutput {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u128,
}

/// Capped output retained on a failed execution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PartialOutput {
    pub stdout: String,
    pub stderr: String,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    /// At least one reader was still running when the failure was reported.
    pub output_incomplete: bool,
    /// Group termination or reaping did not finish within the cleanup grace.
    pub cleanup_incomplete: bool,
    pub cleanup_error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputStream {
    Stdout,
    Stderr,
}

impl std::fmt::Display for OutputStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stdout => write!(f, "stdout"),
            Self::Stderr => write!(f, "stderr"),
        }
    }
}

#[derive(Debug)]
pub enum ExecError {
    Unavailable(String),
    Spawn(io::Error),
    Io(io::Error),
    Timeout {
        timeout_ms: u64,
        duration_ms: u128,
        partial: PartialOutput,
    },
    Cancelled {
        partial: PartialOutput,
    },
    OutputLimit {
        stream: OutputStream,
        limit_bytes: usize,
        partial: PartialOutput,
    },
    Reader {
        source: io::Error,
        stream: OutputStream,
        partial: PartialOutput,
    },
    /// Waiting for the child failed; partial output is still retained.
    Wait {
        source: io::Error,
        partial: PartialOutput,
    },
    /// Child/group cleanup failed after an otherwise successful wait.
    Cleanup {
        source: String,
        partial: PartialOutput,
    },
}

impl std::fmt::Display for ExecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(reason) => write!(f, "execution isolation unavailable: {reason}"),
            Self::Spawn(err) => write!(f, "could not start isolated process: {err}"),
            Self::Io(err) => write!(f, "isolated execution I/O failure: {err}"),
            Self::Timeout { timeout_ms, partial, .. } => write!(
                f,
                "isolated process exceeded its {timeout_ms} ms deadline{}",
                if partial.cleanup_incomplete {
                    " (cleanup incomplete)"
                } else {
                    ""
                }
            ),
            Self::Cancelled { partial } => write!(
                f,
                "isolated execution was cancelled{}",
                if partial.cleanup_incomplete {
                    " (cleanup incomplete)"
                } else {
                    ""
                }
            ),
            Self::OutputLimit {
                stream,
                limit_bytes,
                ..
            } => write!(f, "{stream} exceeded its {limit_bytes}-byte capture limit"),
            Self::Reader { source, stream, .. } => {
                write!(f, "{stream} reader failed: {source}")
            }
            Self::Wait { source, .. } => write!(f, "wait failed: {source}"),
            Self::Cleanup { source, .. } => write!(f, "process cleanup failed: {source}"),
        }
    }
}

impl ExecError {
    pub fn partial(&self) -> Option<&PartialOutput> {
        match self {
            Self::Timeout { partial, .. }
            | Self::Cancelled { partial }
            | Self::OutputLimit { partial, .. }
            | Self::Reader { partial, .. }
            | Self::Wait { partial, .. }
            | Self::Cleanup { partial, .. } => Some(partial),
            _ => None,
        }
    }

    pub fn cleanup_incomplete(&self) -> bool {
        self.partial()
            .is_some_and(|partial| partial.cleanup_incomplete)
    }

    pub fn cleanup_error(&self) -> Option<&str> {
        self.partial()
            .and_then(|partial| partial.cleanup_error.as_deref())
    }
}

impl std::error::Error for ExecError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spawn(err) | Self::Io(err) => Some(err),
            Self::Reader { source, .. } | Self::Wait { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Cooperative cancellation token. Cloneable; cancellation is sticky.
#[derive(Clone, Default)]
pub struct CancellationToken {
    inner: Arc<CancelInner>,
}

#[derive(Default)]
struct CancelInner {
    cancelled: AtomicBool,
    notify: Notify,
}

impl std::fmt::Debug for CancellationToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancellationToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.inner.cancelled.store(true, Ordering::SeqCst);
        self.inner.notify.notify_waiters();
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::SeqCst)
    }

    pub async fn cancelled(&self) {
        loop {
            if self.is_cancelled() {
                return;
            }
            let notified = self.inner.notify.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

/// Injectable execution boundary. Production only ever uses [`SandboxExec`];
/// test-only executors are compiled under `cfg(test)` and injected by trusted
/// callers. Model output never selects an executor.
pub trait Executor: Send + Sync + std::fmt::Debug {
    fn run<'a>(
        &'a self,
        request: &'a ExecRequest<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<ExecOutput, ExecError>> + Send + 'a>>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendInfo {
    pub name: &'static str,
    pub network_denied: bool,
    pub writes_confined: bool,
    pub environment_cleared: bool,
    pub non_root: bool,
    pub resource_limits: bool,
    pub containment_verified: bool,
}

impl BackendInfo {
    pub fn summary(&self) -> String {
        format!(
            "{}: network_denied={} writes_confined={} env_cleared={} non_root={} rlimits={} containment_verified={}",
            self.name,
            self.network_denied,
            self.writes_confined,
            self.environment_cleared,
            self.non_root,
            self.resource_limits,
            self.containment_verified
        )
    }
}

/// The production executor and its process-wide detection cache.
pub struct Isolation;

impl Isolation {
    pub fn shared() -> Result<Arc<SandboxExec>, ExecError> {
        static SHARED: OnceLock<Result<Arc<SandboxExec>, String>> = OnceLock::new();
        match SHARED.get_or_init(|| {
            SandboxExec::detect()
                .map(Arc::new)
                .map_err(|err| err.to_string())
        }) {
            Ok(executor) => Ok(Arc::clone(executor)),
            Err(reason) => Err(ExecError::Unavailable(reason.clone())),
        }
    }

    pub fn production() -> Result<SandboxExec, ExecError> {
        SandboxExec::detect()
    }
}

// ---------------------------------------------------------------------------
// Managed process and supervision
// ---------------------------------------------------------------------------

type BoxRead = Box<dyn AsyncRead + Send + Unpin>;

#[derive(Debug, Clone)]
struct CleanupError {
    message: String,
    /// True only when the leader was successfully reaped despite the error.
    reaped: bool,
}

impl std::fmt::Display for CleanupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

fn cleanup_error(context: &str, err: io::Error) -> CleanupError {
    CleanupError {
        message: format!("{context}: {err}"),
        reaped: false,
    }
}

trait ManagedProcess: Send {
    fn try_wait(&mut self) -> io::Result<Option<i32>>;
    fn take_stdout(&mut self) -> Option<BoxRead>;
    fn take_stderr(&mut self) -> Option<BoxRead>;
    fn leader_reaped(&self) -> bool;
    fn kill_group(&mut self) -> Result<(), CleanupError>;
    fn start_cleanup(&mut self) -> Result<(), CleanupError>;
    fn terminate(&mut self) -> Pin<Box<dyn Future<Output = Result<(), CleanupError>> + Send + '_>>;
    fn reap(&mut self) -> Pin<Box<dyn Future<Output = Result<(), CleanupError>> + Send + '_>>;
}

struct RealProcess {
    child: Child,
    pgid: u32,
    leader_reaped: bool,
}

impl ManagedProcess for RealProcess {
    fn try_wait(&mut self) -> io::Result<Option<i32>> {
        // WNOWAIT: observe exit without reaping so the pgid remains valid while
        // the group is signalled.
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        let rc = unsafe {
            libc::waitid(
                libc::P_PID,
                self.pgid as libc::id_t,
                &mut info,
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            )
        };
        if rc == -1 {
            return Err(io::Error::last_os_error());
        }
        if info.si_pid == 0 {
            return Ok(None);
        }
        let code = if info.si_code == libc::CLD_EXITED {
            info.si_status
        } else {
            128 + info.si_status
        };
        Ok(Some(code))
    }

    fn take_stdout(&mut self) -> Option<BoxRead> {
        self.child
            .stdout
            .take()
            .map(|pipe| Box::new(pipe) as BoxRead)
    }

    fn take_stderr(&mut self) -> Option<BoxRead> {
        self.child
            .stderr
            .take()
            .map(|pipe| Box::new(pipe) as BoxRead)
    }

    fn leader_reaped(&self) -> bool {
        self.leader_reaped
    }

    fn kill_group(&mut self) -> Result<(), CleanupError> {
        if self.leader_reaped || self.pgid == 0 {
            return Ok(());
        }
        let rc = unsafe { libc::kill(-(self.pgid as i32), libc::SIGKILL) };
        if rc == 0 {
            return Ok(());
        }
        let err = io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::ESRCH) {
            Ok(())
        } else {
            Err(cleanup_error("killpg", err))
        }
    }

    fn start_cleanup(&mut self) -> Result<(), CleanupError> {
        if self.leader_reaped {
            return Ok(());
        }
        let mut first: Option<CleanupError> = None;
        if let Err(err) = self.kill_group() {
            first = Some(err);
        }
        if let Err(err) = self.child.start_kill() {
            if first.is_none() {
                first = Some(cleanup_error("start_kill", err));
            }
        }
        match first {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    fn terminate(&mut self) -> Pin<Box<dyn Future<Output = Result<(), CleanupError>> + Send + '_>> {
        Box::pin(async move {
            if self.leader_reaped {
                return Ok(());
            }
            let mut first: Option<CleanupError> = None;
            if let Err(err) = self.kill_group() {
                first = Some(err);
            }
            if let Err(err) = self.child.start_kill() {
                if first.is_none() {
                    first = Some(cleanup_error("start_kill", err));
                }
            }
            if let Err(err) = self.child.wait().await {
                if first.is_none() {
                    first = Some(cleanup_error("wait", err));
                }
            } else {
                self.leader_reaped = true;
            }
            match first {
                Some(mut err) => {
                    err.reaped = self.leader_reaped;
                    Err(err)
                }
                None => Ok(()),
            }
        })
    }

    fn reap(&mut self) -> Pin<Box<dyn Future<Output = Result<(), CleanupError>> + Send + '_>> {
        Box::pin(async move {
            if self.leader_reaped {
                return Ok(());
            }
            self.child
                .wait()
                .await
                .map(|_| {
                    self.leader_reaped = true;
                })
                .map_err(|err| cleanup_error("wait", err))
        })
    }
}

struct Capture {
    bytes: Vec<u8>,
    truncated: bool,
    limit: usize,
}

impl Capture {
    fn new(limit: usize) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            bytes: Vec::new(),
            truncated: false,
            limit,
        }))
    }
}

enum ReaderOutcome {
    Done,
    Limit(OutputStream),
    Io(io::Error),
}

async fn read_limited<R>(
    mut reader: R,
    capture: Arc<Mutex<Capture>>,
    stream: OutputStream,
) -> ReaderOutcome
where
    R: AsyncRead + Unpin,
{
    let mut chunk = [0u8; 4096];
    loop {
        match reader.read(&mut chunk).await {
            Ok(0) => return ReaderOutcome::Done,
            Ok(read) => {
                let mut capture = capture.lock().expect("capture poisoned");
                let remaining = capture.limit.saturating_sub(capture.bytes.len());
                if read > remaining {
                    capture.bytes.extend_from_slice(&chunk[..remaining]);
                    capture.truncated = true;
                    return ReaderOutcome::Limit(stream);
                }
                capture.bytes.extend_from_slice(&chunk[..read]);
            }
            Err(err) => return ReaderOutcome::Io(err),
        }
    }
}

fn capture_snapshot(capture: &Arc<Mutex<Capture>>) -> (String, bool) {
    let capture = capture.lock().expect("capture poisoned");
    (
        String::from_utf8_lossy(&capture.bytes).into_owned(),
        capture.truncated,
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CleanupStatus {
    Pending,
    Complete,
    /// Cleanup failed, but the leader was reaped.
    Failed(String),
    /// Cleanup did not finish; a reaper owns the process.
    Incomplete(String),
}

struct Supervision {
    process: Option<Box<dyn ManagedProcess>>,
    stdout_task: Option<JoinHandle<ReaderOutcome>>,
    stderr_task: Option<JoinHandle<ReaderOutcome>>,
    stdout_capture: Arc<Mutex<Capture>>,
    stderr_capture: Arc<Mutex<Capture>>,
    leader_reaped: bool,
    cleanup: CleanupStatus,
}

impl Supervision {
    fn new(process: Box<dyn ManagedProcess>, limit: usize) -> Self {
        let mut process = process;
        let stdout_capture = Capture::new(limit);
        let stderr_capture = Capture::new(limit);
        let stdout_task = process.take_stdout().map(|reader| {
            tokio::spawn(read_limited(reader, Arc::clone(&stdout_capture), OutputStream::Stdout))
        });
        let stderr_task = process.take_stderr().map(|reader| {
            tokio::spawn(read_limited(reader, Arc::clone(&stderr_capture), OutputStream::Stderr))
        });
        let leader_reaped = process.leader_reaped();
        Self {
            process: Some(process),
            stdout_task,
            stderr_task,
            stdout_capture,
            stderr_capture,
            leader_reaped,
            cleanup: CleanupStatus::Pending,
        }
    }

    fn readers_open(&self) -> bool {
        self.stdout_task.is_some() || self.stderr_task.is_some()
    }

    fn partial(&self, output_incomplete: bool) -> PartialOutput {
        let (stdout, stdout_truncated) = capture_snapshot(&self.stdout_capture);
        let (stderr, stderr_truncated) = capture_snapshot(&self.stderr_capture);
        PartialOutput {
            stdout,
            stderr,
            stdout_truncated,
            stderr_truncated,
            output_incomplete,
            cleanup_incomplete: self.cleanup_incomplete(),
            cleanup_error: self.cleanup_error(),
        }
    }

    fn cleanup_incomplete(&self) -> bool {
        !matches!(self.cleanup, CleanupStatus::Complete)
    }

    fn cleanup_error(&self) -> Option<String> {
        match &self.cleanup {
            CleanupStatus::Failed(error) | CleanupStatus::Incomplete(error) => {
                Some(error.clone())
            }
            _ => None,
        }
    }

    fn cleanup_result(&self) -> (bool, Option<String>) {
        match &self.cleanup {
            CleanupStatus::Complete => (true, None),
            CleanupStatus::Failed(error) | CleanupStatus::Incomplete(error) => {
                (false, Some(error.clone()))
            }
            CleanupStatus::Pending => (false, None),
        }
    }

    fn abort_readers(&mut self) {
        if let Some(task) = self.stdout_task.take() {
            task.abort();
        }
        if let Some(task) = self.stderr_task.take() {
            task.abort();
        }
    }

    fn spawn_reaper(&self, mut process: Box<dyn ManagedProcess>) {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let _ = process.terminate().await;
            });
        }
    }

    async fn terminate(&mut self, grace: Duration) -> (bool, Option<String>) {
        if !matches!(self.cleanup, CleanupStatus::Pending) {
            return self.cleanup_result();
        }
        self.abort_readers();
        if self.process.is_none() {
            self.leader_reaped = true;
            self.cleanup = CleanupStatus::Complete;
            return (true, None);
        }

        let process = self.process.as_mut().expect("process");
        match timeout(grace, process.terminate()).await {
            Ok(Ok(())) => {
                self.leader_reaped = true;
                self.cleanup = CleanupStatus::Complete;
                self.process = None;
                (true, None)
            }
            Ok(Err(err)) => {
                self.leader_reaped = err.reaped;
                if err.reaped {
                    self.cleanup = CleanupStatus::Failed(err.message.clone());
                    self.process = None;
                } else {
                    self.cleanup = CleanupStatus::Incomplete(err.message.clone());
                    let mut process = self.process.take().expect("process");
                    if !self.leader_reaped {
                        let _ = process.start_cleanup();
                    }
                    self.spawn_reaper(process);
                }
                (false, Some(err.message))
            }
            Err(_) => {
                let message = format!("cleanup exceeded its {grace:?} grace");
                self.cleanup = CleanupStatus::Incomplete(message.clone());
                let mut process = self.process.take().expect("process");
                if !self.leader_reaped {
                    let _ = process.start_cleanup();
                }
                self.spawn_reaper(process);
                (false, Some(message))
            }
        }
    }

    async fn finish_success(&mut self, grace: Duration) -> Result<(), CleanupError> {
        if !matches!(self.cleanup, CleanupStatus::Pending) {
            return match self.cleanup {
                CleanupStatus::Complete => Ok(()),
                CleanupStatus::Failed(ref error) | CleanupStatus::Incomplete(ref error) => {
                    Err(CleanupError {
                        message: error.clone(),
                        reaped: self.leader_reaped,
                    })
                }
                CleanupStatus::Pending => unreachable!(),
            };
        }
        let Some(process) = self.process.as_mut() else {
            self.leader_reaped = true;
            self.cleanup = CleanupStatus::Complete;
            return Ok(());
        };

        let signal_error = process.kill_group().err();
        match timeout(grace, process.reap()).await {
            Ok(Ok(())) => {
                self.leader_reaped = true;
                self.process = None;
                match signal_error {
                    Some(error) => {
                        self.cleanup = CleanupStatus::Failed(error.message.clone());
                        Err(CleanupError {
                            message: error.message,
                            reaped: true,
                        })
                    }
                    None => {
                        self.cleanup = CleanupStatus::Complete;
                        Ok(())
                    }
                }
            }
            Ok(Err(error)) => {
                let message = match signal_error {
                    Some(signal) => format!("{signal}; {error}"),
                    None => error.to_string(),
                };
                self.cleanup = CleanupStatus::Incomplete(message.clone());
                let mut process = self.process.take().expect("process");
                if !self.leader_reaped {
                    let _ = process.start_cleanup();
                }
                self.spawn_reaper(process);
                Err(CleanupError {
                    message,
                    reaped: error.reaped,
                })
            }
            Err(_) => {
                let message = format!("reaping exceeded its {grace:?} grace");
                self.cleanup = CleanupStatus::Incomplete(message.clone());
                let mut process = self.process.take().expect("process");
                if !self.leader_reaped {
                    let _ = process.start_cleanup();
                }
                self.spawn_reaper(process);
                Err(CleanupError {
                    message,
                    reaped: false,
                })
            }
        }
    }
}

impl Drop for Supervision {
    fn drop(&mut self) {
        if !matches!(self.cleanup, CleanupStatus::Pending) {
            return;
        }
        self.cleanup = CleanupStatus::Incomplete("future dropped before cleanup completed".into());
        self.abort_readers();
        if let Some(mut process) = self.process.take() {
            if !self.leader_reaped && !process.leader_reaped() {
                let _ = process.start_cleanup();
            }
            self.spawn_reaper(process);
        }
    }
}

enum RunEvent {
    Deadline,
    Cancel,
    Tick,
    Stdout(Option<Result<ReaderOutcome, JoinError>>),
    Stderr(Option<Result<ReaderOutcome, JoinError>>),
}

async fn wait_cancel(cancel: &Option<CancellationToken>) {
    match cancel {
        Some(token) => token.cancelled().await,
        None => std::future::pending().await,
    }
}

async fn join_opt(
    handle: &mut Option<JoinHandle<ReaderOutcome>>,
) -> Option<Result<ReaderOutcome, JoinError>> {
    match handle {
        Some(task) => Some(task.await),
        None => std::future::pending().await,
    }
}

async fn run_bounded(
    mut supervision: Supervision,
    deadline: Instant,
    started: Instant,
    timeout_ms: u64,
    cancel: Option<CancellationToken>,
    limit: usize,
    cleanup_grace: Duration,
) -> Result<ExecOutput, ExecError> {
    let mut exit_code: Option<i32> = None;
    let deadline_timer = sleep(deadline.saturating_duration_since(Instant::now()));
    tokio::pin!(deadline_timer);
    // Give already-spawned readers one scheduling turn so a wait failure can
    // still report bytes that were available before the failure.
    tokio::task::yield_now().await;

    loop {
        if cancel.as_ref().is_some_and(CancellationToken::is_cancelled) {
            let incomplete = supervision.readers_open();
            let mut partial = supervision.partial(incomplete);
            let (complete, detail) = supervision.terminate(cleanup_grace).await;
            partial.cleanup_incomplete = !complete;
            partial.cleanup_error = detail;
            return Err(ExecError::Cancelled { partial });
        }

        if exit_code.is_none() {
            match supervision.process.as_mut().expect("process").try_wait() {
                Ok(Some(code)) => exit_code = Some(code),
                Ok(None) => {}
                Err(err) => {
                    let incomplete = supervision.readers_open();
                    let mut partial = supervision.partial(incomplete);
                    let (complete, detail) = supervision.terminate(cleanup_grace).await;
                    partial.cleanup_incomplete = !complete;
                    partial.cleanup_error = detail;
                    return Err(ExecError::Wait {
                        source: err,
                        partial,
                    });
                }
            }
        }

        if exit_code.is_some() && !supervision.readers_open() {
            if Instant::now() >= deadline {
                let mut partial = supervision.partial(false);
                let (complete, detail) = supervision.terminate(cleanup_grace).await;
                partial.cleanup_incomplete = !complete;
                partial.cleanup_error = detail;
                return Err(ExecError::Timeout {
                    timeout_ms,
                    duration_ms: started.elapsed().as_millis(),
                    partial,
                });
            }
            match supervision.finish_success(cleanup_grace).await {
                Ok(()) => {
                    let partial = supervision.partial(false);
                    if cancel.as_ref().is_some_and(CancellationToken::is_cancelled) {
                        return Err(ExecError::Cancelled { partial });
                    }
                    if Instant::now() >= deadline {
                        return Err(ExecError::Timeout {
                            timeout_ms,
                            duration_ms: started.elapsed().as_millis(),
                            partial,
                        });
                    }
                    let (stdout, _) = capture_snapshot(&supervision.stdout_capture);
                    let (stderr, _) = capture_snapshot(&supervision.stderr_capture);
                    return Ok(ExecOutput {
                        exit_code: exit_code.unwrap_or(-1),
                        stdout,
                        stderr,
                        duration_ms: started.elapsed().as_millis(),
                    });
                }
                Err(err) => {
                    let mut partial = supervision.partial(false);
                    partial.cleanup_incomplete = supervision.cleanup_incomplete();
                    partial.cleanup_error = supervision.cleanup_error();
                    return Err(ExecError::Cleanup {
                        source: err.message,
                        partial,
                    });
                }
            }
        }

        let event = tokio::select! {
            biased;
            _ = wait_cancel(&cancel) => RunEvent::Cancel,
            _ = &mut deadline_timer => RunEvent::Deadline,
            result = join_opt(&mut supervision.stdout_task) => RunEvent::Stdout(result),
            result = join_opt(&mut supervision.stderr_task) => RunEvent::Stderr(result),
            _ = sleep(POLL_INTERVAL) => RunEvent::Tick,
        };

        match event {
            RunEvent::Deadline => {
                let incomplete = supervision.readers_open();
                let mut partial = supervision.partial(incomplete);
                let (complete, detail) = supervision.terminate(cleanup_grace).await;
                partial.cleanup_incomplete = !complete;
                partial.cleanup_error = detail;
                return Err(ExecError::Timeout {
                    timeout_ms,
                    duration_ms: started.elapsed().as_millis(),
                    partial,
                });
            }
            RunEvent::Cancel => {
                let incomplete = supervision.readers_open();
                let mut partial = supervision.partial(incomplete);
                let (complete, detail) = supervision.terminate(cleanup_grace).await;
                partial.cleanup_incomplete = !complete;
                partial.cleanup_error = detail;
                return Err(ExecError::Cancelled { partial });
            }
            RunEvent::Stdout(result) => match result {
                Some(Ok(ReaderOutcome::Done)) => supervision.stdout_task = None,
                Some(Ok(ReaderOutcome::Limit(stream))) => {
                    let incomplete = supervision.readers_open();
                    let mut partial = supervision.partial(incomplete);
                    let (complete, detail) = supervision.terminate(cleanup_grace).await;
                    partial.cleanup_incomplete = !complete;
                    partial.cleanup_error = detail;
                    return Err(ExecError::OutputLimit {
                        stream,
                        limit_bytes: limit,
                        partial,
                    });
                }
                Some(Ok(ReaderOutcome::Io(source))) => {
                    let incomplete = supervision.readers_open();
                    let mut partial = supervision.partial(incomplete);
                    let (complete, detail) = supervision.terminate(cleanup_grace).await;
                    partial.cleanup_incomplete = !complete;
                    partial.cleanup_error = detail;
                    return Err(ExecError::Reader {
                        source,
                        stream: OutputStream::Stdout,
                        partial,
                    });
                }
                Some(Err(err)) => {
                    let incomplete = supervision.readers_open();
                    let mut partial = supervision.partial(incomplete);
                    let (complete, detail) = supervision.terminate(cleanup_grace).await;
                    partial.cleanup_incomplete = !complete;
                    partial.cleanup_error = detail;
                    return Err(ExecError::Reader {
                        source: io::Error::other(err.to_string()),
                        stream: OutputStream::Stdout,
                        partial,
                    });
                }
                None => {}
            },
            RunEvent::Stderr(result) => match result {
                Some(Ok(ReaderOutcome::Done)) => supervision.stderr_task = None,
                Some(Ok(ReaderOutcome::Limit(stream))) => {
                    let incomplete = supervision.readers_open();
                    let mut partial = supervision.partial(incomplete);
                    let (complete, detail) = supervision.terminate(cleanup_grace).await;
                    partial.cleanup_incomplete = !complete;
                    partial.cleanup_error = detail;
                    return Err(ExecError::OutputLimit {
                        stream,
                        limit_bytes: limit,
                        partial,
                    });
                }
                Some(Ok(ReaderOutcome::Io(source))) => {
                    let incomplete = supervision.readers_open();
                    let mut partial = supervision.partial(incomplete);
                    let (complete, detail) = supervision.terminate(cleanup_grace).await;
                    partial.cleanup_incomplete = !complete;
                    partial.cleanup_error = detail;
                    return Err(ExecError::Reader {
                        source,
                        stream: OutputStream::Stderr,
                        partial,
                    });
                }
                Some(Err(err)) => {
                    let incomplete = supervision.readers_open();
                    let mut partial = supervision.partial(incomplete);
                    let (complete, detail) = supervision.terminate(cleanup_grace).await;
                    partial.cleanup_incomplete = !complete;
                    partial.cleanup_error = detail;
                    return Err(ExecError::Reader {
                        source: io::Error::other(err.to_string()),
                        stream: OutputStream::Stderr,
                        partial,
                    });
                }
                None => {}
            },
            RunEvent::Tick => {}
        }
    }
}

// ---------------------------------------------------------------------------
// sandbox-exec backend
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct SandboxExec {
    sandbox_exec: PathBuf,
    python: PathBuf,
}

impl SandboxExec {
    pub fn detect() -> Result<Self, ExecError> {
        if unsafe { libc::geteuid() } == 0 {
            return Err(ExecError::Unavailable(
                "refusing to execute candidates as root".into(),
            ));
        }
        for path in [SANDBOX_EXEC, DEFAULT_PYTHON, CLT_PYTHON, FRAMEWORK_PYTHON] {
            if !Path::new(path).is_file() {
                return Err(ExecError::Unavailable(format!("missing {path}")));
            }
        }

        let sandbox_exec = PathBuf::from(SANDBOX_EXEC);
        let python = PathBuf::from(DEFAULT_PYTHON);
        let mut command = std::process::Command::new(&sandbox_exec);
        command
            .arg("-p")
            .arg(SELF_TEST_PROFILE)
            .arg(&python)
            .arg("-c")
            .arg("print('isolation-ok')")
            .current_dir("/usr")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", "/usr")
            .env("TMPDIR", "/usr")
            .env("LANG", "C")
            .env("LC_ALL", "C")
            .env("TZ", "UTC")
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .env("PYTHONNOUSERSITE", "1");

        let mut child = command.spawn().map_err(|err| {
            ExecError::Unavailable(format!("cannot start {SANDBOX_EXEC}: {err}"))
        })?;
        let started = Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let mut stdout = String::new();
                    let mut stderr = String::new();
                    if let Some(mut pipe) = child.stdout.take() {
                        let _ = pipe.read_to_string(&mut stdout);
                    }
                    if let Some(mut pipe) = child.stderr.take() {
                        let _ = pipe.read_to_string(&mut stderr);
                    }
                    if status.success() && stdout.contains("isolation-ok") {
                        return Ok(Self {
                            sandbox_exec,
                            python,
                        });
                    }
                    return Err(ExecError::Unavailable(format!(
                        "sandbox-exec self-test failed (status {status}): {}",
                        stderr.trim()
                    )));
                }
                Ok(None) => {}
                Err(err) => {
                    return Err(ExecError::Unavailable(format!(
                        "sandbox-exec self-test failed: {err}"
                    )))
                }
            }
            if started.elapsed() > SELF_TEST_TIMEOUT {
                let _ = child.kill();
                return Err(ExecError::Unavailable(
                    "sandbox-exec self-test timed out".into(),
                ));
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    pub fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "sandbox-exec (macOS Seatbelt)",
            network_denied: true,
            writes_confined: true,
            environment_cleared: true,
            non_root: true,
            resource_limits: true,
            containment_verified: false,
        }
    }

    fn profile(&self, workspace: &Path) -> Result<String, ExecError> {
        let raw = workspace.to_string_lossy();
        if raw.contains('"') || raw.contains('\\') || raw.contains('\n') {
            return Err(ExecError::Unavailable(
                "workspace path cannot be represented in an SBPL profile".into(),
            ));
        }
        Ok(format!(
            r#"(version 1)
(deny default)
(allow process-exec (literal "{DEFAULT_PYTHON}") (literal "{CLT_PYTHON}") (literal "{FRAMEWORK_PYTHON}"))
(allow file-read* (subpath "/usr/lib") (subpath "/usr/share/locale") (literal "{DEFAULT_PYTHON}") (subpath "/System/Library/dyld") (subpath "/System/Library/Frameworks") (subpath "/Library/Developer/CommandLineTools/usr/bin") (subpath "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework") (literal "/var/db/xcode_select_link") (subpath "/private/var/db/timezone") (literal "/dev/null") (literal "/dev/urandom") (subpath "{raw}"))
(allow file-write* (subpath "{raw}") (literal "/dev/null"))
(deny network*)"#
        ))
    }

    async fn execute(&self, request: &ExecRequest<'_>) -> Result<ExecOutput, ExecError> {
        let started = Instant::now();
        let deadline = started + Duration::from_millis(request.timeout_ms);

        // Cancellation is checked before the launcher. A pre-cancelled request
        // makes zero launcher calls.
        if request
            .cancel
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(ExecError::Cancelled {
                partial: PartialOutput::default(),
            });
        }

        let profile = self.profile(request.workspace)?;
        let mut command = Command::new(&self.sandbox_exec);
        command
            .arg("-p")
            .arg(profile)
            .arg(&self.python)
            .args(request.args);
        configure_common(&mut command, request.workspace);
        let timeout_ms = request.timeout_ms;
        unsafe {
            command.as_std_mut().pre_exec(move || {
                if libc::setsid() == -1 {
                    return Err(io::Error::last_os_error());
                }
                apply_limits(timeout_ms)
            });
        }

        let child = command.spawn().map_err(ExecError::Spawn)?;
        let pid = child
            .id()
            .ok_or_else(|| ExecError::Io(io::Error::other("child has no pid")))?;
        let process: Box<dyn ManagedProcess> = Box::new(RealProcess { child, pgid: pid, leader_reaped: false });
        let supervision = Supervision::new(process, MAX_CAPTURE_BYTES);

        let output = run_bounded(
            supervision,
            deadline,
            started,
            request.timeout_ms,
            request.cancel.clone(),
            MAX_CAPTURE_BYTES,
            CLEANUP_GRACE,
        )
        .await?;

        if output.exit_code != 0
            && (output.stderr.contains("sandbox_apply")
                || output.stderr.contains("Operation not permitted"))
        {
            return Err(ExecError::Unavailable(format!(
                "sandbox could not be applied: {}",
                output.stderr.trim()
            )));
        }
        Ok(output)
    }
}

impl Executor for SandboxExec {
    fn run<'a>(
        &'a self,
        request: &'a ExecRequest<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<ExecOutput, ExecError>> + Send + 'a>> {
        Box::pin(self.execute(request))
    }
}

const SELF_TEST_PROFILE: &str = r#"(version 1)
(deny default)
(allow process-exec (literal "/usr/bin/python3") (literal "/Library/Developer/CommandLineTools/usr/bin/python3") (literal "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3"))
(allow file-read* (subpath "/usr/lib") (subpath "/usr/share/locale") (literal "/usr/bin/python3") (subpath "/System/Library/dyld") (subpath "/System/Library/Frameworks") (subpath "/Library/Developer/CommandLineTools/usr/bin") (subpath "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework") (literal "/var/db/xcode_select_link") (subpath "/private/var/db/timezone") (literal "/dev/null") (literal "/dev/urandom"))"#;

// ---------------------------------------------------------------------------
// Shared process plumbing
// ---------------------------------------------------------------------------

fn configure_common(command: &mut Command, workspace: &Path) {
    command
        .current_dir(workspace)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", workspace)
        .env("TMPDIR", workspace)
        .env("LANG", "C")
        .env("LC_ALL", "C")
        .env("TZ", "UTC")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("PYTHONNOUSERSITE", "1");
}

fn apply_limits(timeout_ms: u64) -> io::Result<()> {
    let cpu_seconds = (timeout_ms / 1000).saturating_add(1).max(1);
    set_limit(libc::RLIMIT_CPU, cpu_seconds)?;
    set_limit(libc::RLIMIT_FSIZE, MAX_STDOUT_FILESIZE)?;
    set_limit(libc::RLIMIT_NOFILE, MAX_OPEN_FILES)?;
    Ok(())
}

fn set_limit(resource: libc::c_int, value: u64) -> io::Result<()> {
    let limit = libc::rlimit {
        rlim_cur: value as libc::rlim_t,
        rlim_max: value as libc::rlim_t,
    };
    if unsafe { libc::setrlimit(resource, &limit) } != 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::task::{Context, Poll};
    use tokio::io::{AsyncWriteExt, DuplexStream, ReadBuf};

    #[test]
    fn production_detection_reports_availability_without_claiming_containment() {
        match Isolation::production() {
            Ok(executor) => {
                let info = executor.info();
                assert!(info.network_denied);
                assert!(info.writes_confined);
                assert!(!info.containment_verified);
            }
            Err(ExecError::Unavailable(reason)) => {
                assert!(!reason.is_empty(), "unavailability must be explained");
                eprintln!("isolation unavailable on this host: {reason}");
            }
            Err(other) => panic!("unexpected production detection error: {other}"),
        }
    }

    #[test]
    fn sandbox_profile_denies_network_and_confines_writes() {
        let backend = SandboxExec {
            sandbox_exec: PathBuf::from(SANDBOX_EXEC),
            python: PathBuf::from(DEFAULT_PYTHON),
        };
        let profile = backend
            .profile(Path::new("/tmp/h2-workspace"))
            .expect("profile");
        assert!(profile.contains("(deny default)"));
        assert!(profile.contains("(deny network*)"));
        assert!(profile.contains("(subpath \"/tmp/h2-workspace\")"));
        assert!(profile.contains(&format!("(literal \"{DEFAULT_PYTHON}\")")));
        // Portable synthetic home path; no developer-specific absolute path is
        // embedded in the source or test fixture.
        let synthetic_home = std::env::temp_dir().join("h2-synthetic-home");
        let synthetic_home = synthetic_home.to_string_lossy();
        assert!(!profile.contains(synthetic_home.as_ref()));
        assert!(!profile.contains("process-fork"));
        assert!(!profile.contains("(allow signal"));
        assert!(!profile.contains("sysctl-read"));
        assert!(!profile.contains("(allow file-read-metadata)"));
        assert!(!profile.contains("(subpath \"/System/Library\")"));
        assert!(profile.contains("/usr/share/locale"));
        assert_eq!(profile.matches("(allow file-write*").count(), 1);
    }

    #[derive(Default)]
    struct MockState {
        exit: Option<i32>,
        terminate_calls: usize,
        start_cleanup_calls: usize,
        reaps: usize,
        signal_calls: usize,
        leader_reaped: bool,
        fail_signal: bool,
        fail_terminate: bool,
        fail_reap: bool,
        fail_wait: bool,
        terminate_delay: Option<Duration>,
        reap_delay: Option<Duration>,
    }

    struct MockProcess {
        state: Arc<Mutex<MockState>>,
        stdout: Option<BoxRead>,
        stderr: Option<BoxRead>,
    }

    impl ManagedProcess for MockProcess {
        fn try_wait(&mut self) -> io::Result<Option<i32>> {
            let state = self.state.lock().expect("mock");
            if state.fail_wait {
                Err(io::Error::other("wait failed"))
            } else {
                Ok(state.exit)
            }
        }

        fn take_stdout(&mut self) -> Option<BoxRead> {
            self.stdout.take()
        }

        fn take_stderr(&mut self) -> Option<BoxRead> {
            self.stderr.take()
        }

        fn leader_reaped(&self) -> bool {
            self.state.lock().expect("mock").leader_reaped
        }

        fn kill_group(&mut self) -> Result<(), CleanupError> {
            let mut state = self.state.lock().expect("mock");
            if state.leader_reaped {
                return Ok(());
            }
            state.signal_calls += 1;
            if state.fail_signal {
                Err(CleanupError {
                    message: "signal failed".into(),
                    reaped: false,
                })
            } else {
                Ok(())
            }
        }

        fn start_cleanup(&mut self) -> Result<(), CleanupError> {
            let mut state = self.state.lock().expect("mock");
            state.start_cleanup_calls += 1;
            if state.leader_reaped {
                return Ok(());
            }
            state.signal_calls += 1;
            if state.fail_signal {
                Err(CleanupError {
                    message: "signal failed".into(),
                    reaped: false,
                })
            } else {
                Ok(())
            }
        }

        fn terminate(&mut self) -> Pin<Box<dyn Future<Output = Result<(), CleanupError>> + Send + '_>> {
            let state = Arc::clone(&self.state);
            Box::pin(async move {
                let (delay, fail, fail_signal, already_reaped) = {
                    let mut state = state.lock().expect("mock");
                    if state.leader_reaped {
                        return Ok(());
                    }
                    state.terminate_calls += 1;
                    state.exit = Some(0);
                    if !state.leader_reaped {
                        state.signal_calls += 1;
                    }
                    (
                        state.terminate_delay,
                        state.fail_terminate,
                        state.fail_signal,
                        state.leader_reaped,
                    )
                };
                if let Some(delay) = delay {
                    sleep(delay).await;
                }
                if already_reaped {
                    return Ok(());
                }
                if fail {
                    return Err(CleanupError {
                        message: "wait failed".into(),
                        reaped: false,
                    });
                }
                state.lock().expect("mock").leader_reaped = true;
                if fail_signal {
                    Err(CleanupError {
                        message: "signal failed".into(),
                        reaped: true,
                    })
                } else {
                    Ok(())
                }
            })
        }

        fn reap(&mut self) -> Pin<Box<dyn Future<Output = Result<(), CleanupError>> + Send + '_>> {
            let state = Arc::clone(&self.state);
            Box::pin(async move {
                let (fail, already_reaped, delay) = {
                    let mut state = state.lock().expect("mock");
                    state.reaps += 1;
                    (state.fail_reap, state.leader_reaped, state.reap_delay)
                };
                if already_reaped {
                    return Ok(());
                }
                if let Some(delay) = delay {
                    sleep(delay).await;
                }
                if fail {
                    return Err(CleanupError {
                        message: "wait failed".into(),
                        reaped: false,
                    });
                }
                state.lock().expect("mock").leader_reaped = true;
                Ok(())
            })
        }
    }

    fn mock_process() -> (
        MockProcess,
        DuplexStream,
        DuplexStream,
        Arc<Mutex<MockState>>,
    ) {
        let (stdout_reader, stdout_writer) = tokio::io::duplex(4096);
        let (stderr_reader, stderr_writer) = tokio::io::duplex(4096);
        let state = Arc::new(Mutex::new(MockState::default()));
        (
            MockProcess {
                state: Arc::clone(&state),
                stdout: Some(Box::new(stdout_reader) as BoxRead),
                stderr: Some(Box::new(stderr_reader) as BoxRead),
            },
            stdout_writer,
            stderr_writer,
            state,
        )
    }

    async fn bounded(
        process: MockProcess,
        timeout_ms: u64,
        cancel: Option<CancellationToken>,
        limit: usize,
    ) -> Result<ExecOutput, ExecError> {
        let started = Instant::now();
        run_bounded(
            Supervision::new(Box::new(process), limit),
            started + Duration::from_millis(timeout_ms),
            started,
            timeout_ms,
            cancel,
            limit,
            Duration::from_millis(200),
        )
        .await
    }

    #[tokio::test]
    async fn successful_exit_with_closed_pipes_is_bounded_and_reaps() {
        let (process, mut stdout, stderr, state) = mock_process();
        stdout.write_all(b"hello").await.expect("write");
        drop(stdout);
        drop(stderr);
        state.lock().expect("state").exit = Some(0);

        let output = bounded(process, 1_000, None, 1_024)
            .await
            .expect("bounded success");
        assert_eq!(output.stdout, "hello");
        assert_eq!(output.exit_code, 0);
        assert_eq!(state.lock().expect("state").reaps, 1);
    }

    #[tokio::test]
    async fn expired_deadline_at_success_boundary_is_not_success() {
        let (process, stdout, stderr, state) = mock_process();
        drop(stdout);
        drop(stderr);
        state.lock().expect("state").exit = Some(0);

        let started = Instant::now();
        let error = run_bounded(
            Supervision::new(Box::new(process), 1_024),
            started - Duration::from_millis(1),
            started,
            0,
            None,
            1_024,
            Duration::from_millis(200),
        )
        .await
        .expect_err("expired deadline must fail");
        assert!(matches!(error, ExecError::Timeout { .. }));
        assert_eq!(state.lock().expect("state").terminate_calls, 1);
    }

    #[tokio::test]
    async fn inherited_pipe_times_out_and_retains_partial_output() {
        let (process, mut stdout, _stderr, state) = mock_process();
        stdout.write_all(b"partial").await.expect("write");
        state.lock().expect("state").exit = Some(0);

        let started = Instant::now();
        let error = run_bounded(
            Supervision::new(Box::new(process), 1_024),
            started + Duration::from_millis(80),
            started,
            80,
            None,
            1_024,
            Duration::from_millis(200),
        )
        .await
        .expect_err("inherited pipe must not succeed");
        match error {
            ExecError::Timeout { partial, .. } => {
                assert_eq!(partial.stdout, "partial");
                assert!(partial.output_incomplete);
                assert!(!partial.stdout_truncated);
            }
            other => panic!("unexpected error: {other}"),
        }
        assert_eq!(state.lock().expect("state").terminate_calls, 1);
    }

    #[tokio::test]
    async fn output_flood_is_typed_with_truncated_partial_output() {
        let (process, mut stdout, _stderr, state) = mock_process();
        stdout
            .write_all(&vec![b'x'; 512])
            .await
            .expect("write");
        drop(stdout);

        let error = bounded(process, 1_000, None, 64)
            .await
            .expect_err("flood");
        match error {
            ExecError::OutputLimit {
                stream: OutputStream::Stdout,
                limit_bytes: 64,
                partial,
            } => {
                assert_eq!(partial.stdout.len(), 64);
                assert!(partial.stdout_truncated);
                assert!(partial.output_incomplete);
            }
            other => panic!("unexpected error: {other}"),
        }
        assert_eq!(state.lock().expect("state").terminate_calls, 1);
    }

    #[tokio::test]
    async fn stderr_flood_is_typed() {
        let (process, _stdout, mut stderr, state) = mock_process();
        stderr
            .write_all(&vec![b'y'; 512])
            .await
            .expect("write");
        drop(stderr);

        let error = bounded(process, 1_000, None, 64)
            .await
            .expect_err("flood");
        assert!(matches!(
            error,
            ExecError::OutputLimit {
                stream: OutputStream::Stderr,
                limit_bytes: 64,
                ..
            }
        ));
        assert_eq!(state.lock().expect("state").terminate_calls, 1);
    }

    #[tokio::test]
    async fn cancellation_before_launch_and_during_execution_cleans_once() {
        let (process, _stdout, _stderr, state) = mock_process();
        let token = CancellationToken::new();
        token.cancel();
        let error = bounded(process, 5_000, Some(token), 1_024)
            .await
            .expect_err("cancelled");
        assert!(matches!(error, ExecError::Cancelled { .. }));
        assert_eq!(state.lock().expect("state").terminate_calls, 1);

        let (process, _stdout, _stderr, state) = mock_process();
        let token = CancellationToken::new();
        let task_token = token.clone();
        let started = Instant::now();
        let handle = tokio::spawn(async move {
            run_bounded(
                Supervision::new(Box::new(process), 1_024),
                started + Duration::from_secs(5),
                started,
                5_000,
                Some(task_token),
                1_024,
                Duration::from_millis(200),
            )
            .await
        });
        sleep(Duration::from_millis(20)).await;
        token.cancel();
        assert!(matches!(
            handle.await.expect("task"),
            Err(ExecError::Cancelled { .. })
        ));
        assert_eq!(state.lock().expect("state").terminate_calls, 1);
    }

    #[tokio::test]
    async fn cancellation_retains_partial_output() {
        let (process, mut stdout, _stderr, state) = mock_process();
        stdout.write_all(b"part").await.expect("write");
        let token = CancellationToken::new();
        let task_token = token.clone();
        let started = Instant::now();
        let handle = tokio::spawn(async move {
            run_bounded(
                Supervision::new(Box::new(process), 1_024),
                started + Duration::from_secs(5),
                started,
                5_000,
                Some(task_token),
                1_024,
                Duration::from_millis(200),
            )
            .await
        });
        sleep(Duration::from_millis(30)).await;
        token.cancel();
        match handle.await.expect("task") {
            Err(ExecError::Cancelled { partial }) => {
                assert_eq!(partial.stdout, "part");
                assert!(!partial.stdout_truncated);
            }
            other => panic!("unexpected result: {other:?}"),
        }
        assert_eq!(state.lock().expect("state").terminate_calls, 1);
    }

    #[tokio::test]
    async fn future_drop_before_poll_starts_cleanup() {
        let (process, _stdout, _stderr, state) = mock_process();
        let started = Instant::now();
        let future = run_bounded(
            Supervision::new(Box::new(process), 1_024),
            started + Duration::from_secs(5),
            started,
            5_000,
            None,
            1_024,
            Duration::from_millis(200),
        );
        drop(future);
        sleep(Duration::from_millis(30)).await;
        assert!(state.lock().expect("state").start_cleanup_calls >= 1);
    }

    #[tokio::test]
    async fn future_abort_starts_cleanup_and_spawns_reaper() {
        let (process, _stdout, _stderr, state) = mock_process();
        let started = Instant::now();
        let handle = tokio::spawn(async move {
            run_bounded(
                Supervision::new(Box::new(process), 1_024),
                started + Duration::from_secs(5),
                started,
                5_000,
                None,
                1_024,
                Duration::from_millis(200),
            )
            .await
        });
        sleep(Duration::from_millis(20)).await;
        handle.abort();
        let _ = handle.await;
        sleep(Duration::from_millis(50)).await;
        let state = state.lock().expect("state");
        assert!(state.start_cleanup_calls >= 1);
        assert!(state.terminate_calls >= 1);
    }

    #[tokio::test]
    async fn stalled_cleanup_is_reported_incomplete() {
        let (process, _stdout, _stderr, state) = mock_process();
        state.lock().expect("state").terminate_delay = Some(Duration::from_secs(10));
        // Force a timeout quickly; cleanup grace is short.
        let started = Instant::now();
        let error = run_bounded(
            Supervision::new(Box::new(process), 1_024),
            started + Duration::from_millis(20),
            started,
            20,
            None,
            1_024,
            Duration::from_millis(20),
        )
        .await
        .expect_err("timeout");
        match error {
            ExecError::Timeout { partial, .. } => {
                assert!(partial.cleanup_incomplete);
                assert!(partial.cleanup_error.is_some());
            }
            other => panic!("unexpected error: {other}"),
        }
        assert!(state.lock().expect("state").terminate_calls >= 1);
    }

    #[tokio::test]
    async fn signal_and_reap_failures_are_typed() {
        let (process, stdout, stderr, state) = mock_process();
        drop(stdout);
        drop(stderr);
        state.lock().expect("state").exit = Some(0);
        state.lock().expect("state").fail_reap = true;
        let error = bounded(process, 1_000, None, 1_024)
            .await
            .expect_err("reap failure");
        assert!(matches!(error, ExecError::Cleanup { .. }));
        assert_eq!(state.lock().expect("state").reaps, 1);

        let (process, stdout, stderr, state) = mock_process();
        drop(stdout);
        drop(stderr);
        state.lock().expect("state").exit = Some(0);
        state.lock().expect("state").fail_signal = true;
        let error = bounded(process, 1_000, None, 1_024)
            .await
            .expect_err("signal failure");
        match error {
            ExecError::Cleanup { .. } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    #[tokio::test]
    async fn cleanup_is_idempotent() {
        let (process, _stdout, _stderr, state) = mock_process();
        let mut supervision = Supervision::new(Box::new(process), 64);
        let (first, _) = supervision
            .terminate(Duration::from_millis(50))
            .await;
        let (second, _) = supervision
            .terminate(Duration::from_millis(50))
            .await;
        assert!(first && second);
        assert_eq!(state.lock().expect("state").terminate_calls, 1);
        // Recreate because the prior mock was moved.
        let (process, _stdout, _stderr, state) = mock_process();
        let mut supervision = Supervision::new(Box::new(process), 64);
        let _ = supervision.finish_success(Duration::from_millis(50)).await;
        assert_eq!(state.lock().expect("state").reaps, 1);
    }

    #[tokio::test]
    async fn successful_reap_that_never_completes_is_reported_incomplete() {
        let (process, stdout, stderr, state) = mock_process();
        drop(stdout);
        drop(stderr);
        state.lock().expect("state").exit = Some(0);
        state.lock().expect("state").reap_delay = Some(Duration::from_secs(10));

        let started = Instant::now();
        let error = run_bounded(
            Supervision::new(Box::new(process), 1_024),
            started + Duration::from_secs(1),
            started,
            1_000,
            None,
            1_024,
            Duration::from_millis(50),
        )
        .await
        .expect_err("stalled reap");
        match error {
            ExecError::Cleanup { partial, .. } => {
                assert!(partial.cleanup_incomplete);
                assert!(partial.cleanup_error.is_some());
            }
            other => panic!("unexpected error: {other}"),
        }
        assert!(state.lock().expect("state").start_cleanup_calls >= 1);
    }

    #[tokio::test]
    async fn cancellation_during_successful_cleanup_returns_cancelled() {
        let (process, stdout, stderr, state) = mock_process();
        drop(stdout);
        drop(stderr);
        state.lock().expect("state").exit = Some(0);
        state.lock().expect("state").reap_delay = Some(Duration::from_millis(80));

        let token = CancellationToken::new();
        let task_token = token.clone();
        let started = Instant::now();
        let handle = tokio::spawn(async move {
            run_bounded(
                Supervision::new(Box::new(process), 1_024),
                started + Duration::from_secs(5),
                started,
                5_000,
                Some(task_token),
                1_024,
                Duration::from_millis(300),
            )
            .await
        });
        sleep(Duration::from_millis(20)).await;
        token.cancel();
        match handle.await.expect("task") {
            Err(ExecError::Cancelled { partial }) => {
                assert!(!partial.cleanup_incomplete);
            }
            other => panic!("unexpected result: {other:?}"),
        }
        assert!(state.lock().expect("state").leader_reaped);
    }

    #[tokio::test]
    async fn deadline_during_successful_cleanup_returns_timeout() {
        let (process, stdout, stderr, state) = mock_process();
        drop(stdout);
        drop(stderr);
        state.lock().expect("state").exit = Some(0);
        state.lock().expect("state").reap_delay = Some(Duration::from_millis(80));

        let started = Instant::now();
        let error = run_bounded(
            Supervision::new(Box::new(process), 1_024),
            started + Duration::from_millis(30),
            started,
            30,
            None,
            1_024,
            Duration::from_millis(300),
        )
        .await
        .expect_err("deadline during cleanup");
        match error {
            ExecError::Timeout { partial, .. } => {
                assert!(!partial.cleanup_incomplete);
            }
            other => panic!("unexpected error: {other}"),
        }
        assert!(state.lock().expect("state").leader_reaped);
    }

    #[tokio::test]
    async fn signal_error_with_successful_reap_is_retained_and_not_resignaled() {
        let (process, stdout, stderr, state) = mock_process();
        drop(stdout);
        drop(stderr);
        state.lock().expect("state").fail_signal = true;
        state.lock().expect("state").exit = Some(0);

        let mut supervision = Supervision::new(Box::new(process), 1_024);
        let (first_complete, first_error) = supervision
            .terminate(Duration::from_millis(100))
            .await;
        assert!(!first_complete);
        assert!(first_error.as_deref().is_some_and(|e| e.contains("signal")));
        assert!(state.lock().expect("state").leader_reaped);
        let signals_after_first = state.lock().expect("state").signal_calls;
        assert_eq!(signals_after_first, 1);

        let (second_complete, second_error) = supervision
            .terminate(Duration::from_millis(100))
            .await;
        assert!(!second_complete);
        assert_eq!(second_error, first_error);
        assert_eq!(state.lock().expect("state").signal_calls, signals_after_first);

        drop(supervision);
        assert_eq!(state.lock().expect("state").signal_calls, signals_after_first);
    }

    #[tokio::test]
    async fn repeated_incomplete_cleanup_keeps_the_error() {
        let (process, _stdout, _stderr, state) = mock_process();
        state.lock().expect("state").terminate_delay = Some(Duration::from_secs(10));

        let mut supervision = Supervision::new(Box::new(process), 1_024);
        let (first_complete, first_error) = supervision
            .terminate(Duration::from_millis(20))
            .await;
        assert!(!first_complete);
        let second = supervision
            .terminate(Duration::from_millis(20))
            .await;
        assert!(!second.0);
        assert_eq!(second.1, first_error);
        assert!(state.lock().expect("state").terminate_calls >= 1);
    }

    #[tokio::test]
    async fn try_wait_failure_retains_partial_output() {
        let (process, mut stdout, _stderr, state) = mock_process();
        stdout.write_all(b"partial").await.expect("write");
        state.lock().expect("state").fail_wait = true;

        let error = bounded(process, 1_000, None, 1_024)
            .await
            .expect_err("wait failure");
        match error {
            ExecError::Wait { partial, .. } => {
                assert_eq!(partial.stdout, "partial");
                assert!(partial.output_incomplete);
            }
            other => panic!("unexpected error: {other}"),
        }
        assert!(state.lock().expect("state").terminate_calls >= 1);
    }

    #[tokio::test]
    async fn reader_io_error_is_typed_with_partial() {
        struct Faulty {
            emitted: bool,
        }
        impl AsyncRead for Faulty {
            fn poll_read(
                mut self: Pin<&mut Self>,
                _cx: &mut Context<'_>,
                buf: &mut ReadBuf<'_>,
            ) -> Poll<io::Result<()>> {
                if !self.emitted {
                    buf.put_slice(b"partial");
                    self.emitted = true;
                    Poll::Ready(Ok(()))
                } else {
                    Poll::Ready(Err(io::Error::other("reader boom")))
                }
            }
        }

        let (process, _stdout, _stderr, state) = mock_process();
        let mut process = process;
        process.stdout = Some(Box::new(Faulty { emitted: false }) as BoxRead);

        let error = bounded(process, 1_000, None, 1_024)
            .await
            .expect_err("reader error");
        match error {
            ExecError::Reader {
                stream: OutputStream::Stdout,
                partial,
                ..
            } => {
                assert_eq!(partial.stdout, "partial");
                assert!(partial.output_incomplete);
            }
            other => panic!("unexpected error: {other}"),
        }
        assert_eq!(state.lock().expect("state").terminate_calls, 1);
    }

    #[tokio::test]
    async fn pre_cancelled_launcher_makes_zero_launcher_calls() {
        let dir = tempfile::tempdir().expect("dir");
        let marker = dir.path().join("launched");
        let launcher = dir.path().join("launcher.sh");
        std::fs::write(
            &launcher,
            format!("#!/bin/sh\ntouch '{}'\n", marker.display()),
        )
        .expect("launcher");
        let backend = SandboxExec {
            sandbox_exec: launcher,
            python: PathBuf::from(DEFAULT_PYTHON),
        };
        let token = CancellationToken::new();
        token.cancel();
        let request = ExecRequest {
            workspace: dir.path(),
            program: Path::new(DEFAULT_PYTHON),
            args: &[],
            timeout_ms: 1_000,
            cancel: Some(token),
        };
        let error = backend.execute(&request).await.expect_err("cancelled");
        assert!(matches!(error, ExecError::Cancelled { .. }));
        assert!(!marker.exists(), "launcher must not be invoked");
    }

    /// Containment checks must run only on a host where `sandbox-exec` can
    /// actually apply. In this nested environment they are ignored and are not
    /// proof. Unavailable containment is unverified, never passed.
    #[tokio::test]
    #[ignore = "requires an unsandboxed host; sandbox-exec cannot be nested"]
    async fn sandbox_exec_contains_reads_writes_sentinel_and_network_policy() {
        let isolation = Isolation::shared().expect("isolation available");
        let workspace = tempfile::tempdir().expect("workspace");
        let canary_dir = tempfile::tempdir().expect("canaries");
        let canary_read = canary_dir.path().join("read-canary.txt");
        let canary_write = canary_dir.path().join("write-canary.txt");
        std::fs::write(&canary_read, "canary").expect("read canary");
        std::fs::write(&canary_write, "intact").expect("write canary");
        std::fs::write(
            workspace.path().join("canaries.txt"),
            format!(
                "{}\n{}\n",
                canary_read.display(),
                canary_write.display()
            ),
        )
        .expect("canary list");

        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("listener");
        let port = listener.local_addr().expect("addr").port();
        // Control connection from outside the candidate sandbox: proves the
        // endpoint is reachable before a denial is interpreted as policy.
        std::net::TcpStream::connect(("127.0.0.1", port))
            .expect("control connection")
            .shutdown(std::net::Shutdown::Both)
            .expect("control shutdown");

        let script = workspace.path().join("check.py");
        std::fs::write(
            &script,
            r#"import os, pathlib, socket, sys

def denied(fn):
    try:
        fn()
        return False
    except PermissionError:
        return True
    except Exception:
        return False

read_canary, write_canary = pathlib.Path("canaries.txt").read_text().splitlines()
host = denied(lambda: pathlib.Path(read_canary).read_text())
outside = denied(lambda: pathlib.Path(write_canary).write_text("tampered"))
sentinel = os.environ.get("HARNESS2_SENTINEL_PROBE") is None
try:
    sock = socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1)
    sock.close()
    network = "connected"
except PermissionError:
    network = "policy"
except OSError as exc:
    network = f"other:{exc.errno}"
print(
    f"host={'denied' if host else 'allowed'} "
    f"outside={'denied' if outside else 'allowed'} "
    f"sentinel={'absent' if sentinel else 'present'} network={network}"
)
"#,
        )
        .expect("script");

        std::env::set_var("HARNESS2_SENTINEL_PROBE", "leaked");
        let args = vec![script.to_string_lossy().into_owned(), port.to_string()];
        let request = ExecRequest {
            workspace: workspace.path(),
            program: Path::new(DEFAULT_PYTHON),
            args: &args,
            timeout_ms: 5_000,
            cancel: None,
        };
        let output = isolation.run(&request).await.expect("contained run");
        std::env::remove_var("HARNESS2_SENTINEL_PROBE");

        assert_eq!(
            output.stdout.trim(),
            "host=denied outside=denied sentinel=absent network=policy"
        );
        assert_eq!(
            std::fs::read_to_string(&canary_write).expect("canary"),
            "intact"
        );
    }

    #[tokio::test]
    async fn launch_failure_is_typed() {
        let backend = SandboxExec {
            sandbox_exec: PathBuf::from("/nonexistent/sandbox-exec"),
            python: PathBuf::from(DEFAULT_PYTHON),
        };
        let workspace = tempfile::tempdir().expect("workspace");
        let request = ExecRequest {
            workspace: workspace.path(),
            program: Path::new(DEFAULT_PYTHON),
            args: &[],
            timeout_ms: 100,
            cancel: None,
        };
        let error = backend.execute(&request).await.expect_err("launch");
        assert!(matches!(error, ExecError::Spawn(_)), "{error}");
    }
}
