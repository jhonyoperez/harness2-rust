//! Test-only host executor.
//!
//! Compiled only for this crate's unit tests. It provides **no isolation** and
//! must never be used for candidate execution outside tests. Production code
//! uses `Isolation::shared()` (macOS `sandbox-exec`).

use std::future::Future;
use std::io;
use std::pin::Pin;
use std::process::Stdio;
use std::time::{Duration, Instant};

use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::task::JoinHandle;
use tokio::time::timeout;

use crate::execution::{ExecError, ExecOutput, ExecRequest, Executor, PartialOutput};

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct HostExecutor;

impl Executor for HostExecutor {
    fn run<'a>(
        &'a self,
        request: &'a ExecRequest<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<ExecOutput, ExecError>> + Send + 'a>> {
        Box::pin(run_host(request))
    }
}

async fn run_host(request: &ExecRequest<'_>) -> Result<ExecOutput, ExecError> {
    let mut command = Command::new(request.program);
    command
        .args(request.args)
        .current_dir(request.workspace)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", request.workspace)
        .env("TMPDIR", request.workspace)
        .env("LANG", "C")
        .env("LC_ALL", "C")
        .env("TZ", "UTC")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("PYTHONNOUSERSITE", "1");

    run_command(command, request.timeout_ms).await
}

async fn run_command(mut command: Command, timeout_ms: u64) -> Result<ExecOutput, ExecError> {
    let started = Instant::now();
    let mut child = command.spawn().map_err(ExecError::Spawn)?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ExecError::Io(io::Error::other("stdout pipe unavailable")))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| ExecError::Io(io::Error::other("stderr pipe unavailable")))?;
    let stdout_task = tokio::spawn(read_stream(stdout));
    let stderr_task = tokio::spawn(read_stream(stderr));

    match timeout(Duration::from_millis(timeout_ms), child.wait()).await {
        Ok(status) => {
            let status = status.map_err(ExecError::Io)?;
            Ok(ExecOutput {
                exit_code: status.code().unwrap_or(-1),
                stdout: join_reader(stdout_task).await,
                stderr: join_reader(stderr_task).await,
                duration_ms: started.elapsed().as_millis(),
            })
        }
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            let stdout = join_reader_within(stdout_task, Duration::from_secs(1)).await;
            let stderr = join_reader_within(stderr_task, Duration::from_secs(1)).await;
            Err(ExecError::Timeout {
                timeout_ms,
                duration_ms: started.elapsed().as_millis(),
                partial: PartialOutput {
                    stdout,
                    stderr,
                    output_incomplete: true,
                    ..PartialOutput::default()
                },
            })
        }
    }
}

async fn read_stream<R>(mut reader: R) -> io::Result<String>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer).await?;
    Ok(String::from_utf8_lossy(&buffer).into_owned())
}

async fn join_reader(task: JoinHandle<io::Result<String>>) -> String {
    match task.await {
        Ok(Ok(text)) => text,
        _ => String::new(),
    }
}

async fn join_reader_within(task: JoinHandle<io::Result<String>>, limit: Duration) -> String {
    match timeout(limit, task).await {
        Ok(Ok(Ok(text))) => text,
        _ => String::new(),
    }
}
