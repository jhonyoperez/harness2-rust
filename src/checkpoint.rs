//! Versioned, atomic run checkpoints and fail-closed recovery.
//!
//! A checkpoint stores only runtime-owned state: run/attempt IDs, phase,
//! accepted surface and its digest, attempts and evaluation evidence, consumed
//! accounting, cancellation state, and cleanup metadata. Model credentials,
//! environment variables, and expected fixture answers are never serialized
//! automatically. Accepted/candidate source and partial stdout/stderr **are**
//! stored verbatim and can contain sensitive information.
//!
//! Writes are temp-file + `fsync` + rename + directory `fsync`; a crash leaves
//! either the previous commit or the new one, never a torn current file. The
//! file is an envelope containing the payload and its SHA-256, so truncation or
//! accidental corruption fails closed. SHA-256 here is **corruption detection,
//! not authentication**: a writer able to rewrite the file can also rewrite the
//! checksum. There is no fallback to a different run or tree.
//!
//! Recovery rules:
//!
//! * `Completed` runs are not executed again.
//! * An admitted or in-flight attempt, incomplete/failed cleanup, or
//!   `CancelledCleanupIncomplete` is `Blocked` pending explicit
//!   [`CheckpointStore::reconcile_abandon`].
//! * A pre-admission `Running` checkpoint is `Ready`.
//! * A missing, corrupt, or version-mismatched checkpoint is an error; no valid
//!   state means fail closed.
//!
//! A per-run lock file admits only one recovery owner. Stale locks are **not**
//! removed automatically; that requires explicit operator reconciliation.
//! Checkpoints never store PIDs and recovery never signals saved processes.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::evaluator::EvalReport;
use crate::execution::PartialOutput;
use crate::provider::{ProviderBinding, Usage};

pub const CHECKPOINT_VERSION: u32 = 3;
const CURRENT_FILE: &str = "checkpoint.json";
const LOCK_FILE: &str = "checkpoint.lock";

static OWNER_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub enum CheckpointError {
    Io(io::Error),
    Json(serde_json::Error),
    NotFound(PathBuf),
    AlreadyExists(PathBuf),
    Locked(PathBuf),
    Corrupt(String),
    InvalidState(String),
    Version(u32),
    /// The write definitely did not commit; the in-memory state is unchanged.
    Uncommitted(String),
    /// Rename succeeded but durability confirmation failed; the disk state may
    /// be new. Admissions and all other mutations are blocked until
    /// [`CheckpointStore::reconcile_from_disk`] resolves it.
    Uncertain(String),
    /// The store is in an uncertain state and must be reconciled from disk.
    UncertainState(String),
    /// The durable provider/model/config identity does not match this run.
    ConfigurationMismatch(String),
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "checkpoint I/O error: {err}"),
            Self::Json(err) => write!(f, "checkpoint JSON error: {err}"),
            Self::NotFound(path) => write!(f, "checkpoint not found: {}", path.display()),
            Self::AlreadyExists(path) => {
                write!(f, "checkpoint already exists: {}", path.display())
            }
            Self::Locked(path) => write!(
                f,
                "checkpoint is owned by another recovery process: {}",
                path.display()
            ),
            Self::Corrupt(reason) => write!(f, "checkpoint is corrupt: {reason}"),
            Self::InvalidState(reason) => write!(f, "checkpoint state is invalid: {reason}"),
            Self::Version(version) => write!(f, "unsupported checkpoint version {version}"),
            Self::Uncommitted(reason) => write!(f, "checkpoint write was not committed: {reason}"),
            Self::Uncertain(reason) => write!(
                f,
                "checkpoint write outcome is uncertain; reconcile from disk: {reason}"
            ),
            Self::UncertainState(reason) => {
                write!(f, "checkpoint store requires disk reconciliation: {reason}")
            }
            Self::ConfigurationMismatch(reason) => {
                write!(f, "checkpoint configuration mismatch: {reason}")
            }
        }
    }
}

impl std::error::Error for CheckpointError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Json(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for CheckpointError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<serde_json::Error> for CheckpointError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Idle,
    Admitted,
    Generation,
    Probe,
    Evaluation,
    Promotion,
    Blocked,
    Terminal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptPhase {
    Admitted,
    Generating,
    ProbeInFlight,
    ProbeComplete,
    EvaluationInFlight,
    Evaluated,
    Promoted,
    Rejected,
    Cancelled,
    Abandoned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Blocked,
    Completed,
    Cancelled,
    CancelledCleanupIncomplete,
    Exhausted,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CancellationState {
    #[default]
    NotRequested,
    Requested,
    Observed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CleanupState {
    #[default]
    NotRun,
    Complete,
    Incomplete { error: String },
    Failed { error: String },
}

impl CleanupState {
    pub fn is_incomplete(&self) -> bool {
        matches!(self, Self::Incomplete { .. } | Self::Failed { .. })
    }

    pub fn error(&self) -> Option<&str> {
        match self {
            Self::Incomplete { error } | Self::Failed { error } => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartialSnapshot {
    pub stdout: String,
    pub stderr: String,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    pub output_incomplete: bool,
    pub cleanup_incomplete: bool,
    pub cleanup_error: Option<String>,
}

impl From<&PartialOutput> for PartialSnapshot {
    fn from(partial: &PartialOutput) -> Self {
        Self {
            stdout: partial.stdout.clone(),
            stderr: partial.stderr.clone(),
            stdout_truncated: partial.stdout_truncated,
            stderr_truncated: partial.stderr_truncated,
            output_incomplete: partial.output_incomplete,
            cleanup_incomplete: partial.cleanup_incomplete,
            cleanup_error: partial.cleanup_error.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaseEvidence {
    pub name: String,
    pub passed: bool,
    pub exit_code: i32,
    pub actual_sha256: String,
    pub stderr_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationEvidence {
    pub passed: bool,
    /// Exact candidate surface this evidence applies to.
    pub candidate_sha256: String,
    /// Digest of the evaluator configuration (script, fixtures, limits, policy).
    pub evaluator_config_sha256: String,
    pub case_count: usize,
    pub failure_count: usize,
    pub cases: Vec<CaseEvidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconciliationConfirmation {
    pub operator: String,
    pub note: String,
    /// The caller confirms no execution outcome remains unknown.
    pub execution_uncertainty_resolved: bool,
    /// The caller confirms cleanup was independently verified.
    pub cleanup_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reconciliation {
    pub operator: String,
    pub note: String,
    pub execution_uncertainty_resolved: bool,
    pub cleanup_verified: bool,
    pub resolved_attempt_id: Option<String>,
}

impl EvaluationEvidence {
    /// Build evidence without expected answers or raw fixture contents.
    pub fn from_report(
        report: &EvalReport,
        candidate_sha256: impl Into<String>,
        evaluator_config_sha256: impl Into<String>,
    ) -> Self {
        let cases = report
            .outcomes
            .iter()
            .map(|case| CaseEvidence {
                name: case.name.clone(),
                passed: case.passed,
                exit_code: case.exit_code,
                actual_sha256: sha256_hex(case.actual.as_bytes()),
                stderr_sha256: sha256_hex(case.stderr.as_bytes()),
            })
            .collect();
        Self {
            passed: report.passed(),
            candidate_sha256: candidate_sha256.into(),
            evaluator_config_sha256: evaluator_config_sha256.into(),
            case_count: report.outcomes.len(),
            failure_count: report.failures(),
            cases,
        }
    }

    fn validate(&self) -> Result<(), CheckpointError> {
        if self.candidate_sha256.is_empty() || self.evaluator_config_sha256.is_empty() {
            return Err(CheckpointError::InvalidState(
                "evaluation evidence is not bound to candidate and evaluator".into(),
            ));
        }
        if self.case_count == 0 || self.cases.len() != self.case_count {
            return Err(CheckpointError::InvalidState(
                "evaluation evidence has inconsistent case counts".into(),
            ));
        }
        let failures = self.cases.iter().filter(|case| !case.passed).count();
        if failures != self.failure_count {
            return Err(CheckpointError::InvalidState(
                "evaluation failure count does not match cases".into(),
            ));
        }
        if self.passed != (self.failure_count == 0) {
            return Err(CheckpointError::InvalidState(
                "evaluation pass flag does not match cases".into(),
            ));
        }
        if self.cases.iter().any(|case| case.name.trim().is_empty()) {
            return Err(CheckpointError::InvalidState(
                "evaluation evidence contains an unnamed case".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptCheckpoint {
    pub attempt_id: String,
    pub index: usize,
    pub admitted: bool,
    pub charged: bool,
    pub phase: AttemptPhase,
    pub candidate_sha256: Option<String>,
    pub candidate: Option<BTreeMap<String, String>>,
    pub evaluation: Option<EvaluationEvidence>,
    /// Evaluator configuration digest bound before evaluation starts.
    pub evaluator_config_sha256: Option<String>,
    pub failure: Option<String>,
    pub cleanup: CleanupState,
    pub partial: Option<PartialSnapshot>,
    /// Non-secret provider/model/config identity for this attempt.
    pub provider: Option<ProviderBinding>,
    /// Actual, estimated, or unknown provider usage for this attempt.
    pub usage: Option<Usage>,
    /// Explicit operator resolution, distinct from the original failure.
    pub reconciliation: Option<Reconciliation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checkpoint {
    pub version: u32,
    pub run_id: String,
    pub status: RunStatus,
    pub phase: Phase,
    pub accepted_surface: BTreeMap<String, String>,
    pub accepted_surface_sha256: String,
    /// Total admissions; this is the durable accounting record.
    pub consumed: usize,
    pub attempts: Vec<AttemptCheckpoint>,
    pub active_attempt: Option<usize>,
    pub cancellation: CancellationState,
    pub cleanup: CleanupState,
    /// Persisted reason for a blocked run, including uncertain persistence.
    pub block_reason: Option<String>,
    /// Run-level explicit reconciliation (when no active attempt exists).
    pub reconciliation: Option<Reconciliation>,
    /// Non-secret provider/model/config identity for the run.
    pub provider: Option<ProviderBinding>,
}

impl Checkpoint {
    pub fn new(run_id: impl Into<String>, surface: &BTreeMap<PathBuf, String>) -> Self {
        let accepted_surface = surface_map(surface);
        let accepted_surface_sha256 = surface_digest(&accepted_surface);
        Self {
            version: CHECKPOINT_VERSION,
            run_id: run_id.into(),
            status: RunStatus::Running,
            phase: Phase::Idle,
            accepted_surface,
            accepted_surface_sha256,
            consumed: 0,
            attempts: Vec::new(),
            active_attempt: None,
            cancellation: CancellationState::NotRequested,
            cleanup: CleanupState::NotRun,
            block_reason: None,
            reconciliation: None,
            provider: None,
        }
    }

    pub fn surface(&self) -> BTreeMap<PathBuf, String> {
        self.accepted_surface
            .iter()
            .map(|(path, content)| (PathBuf::from(path), content.clone()))
            .collect()
    }

    pub fn active(&self) -> Option<&AttemptCheckpoint> {
        self.active_attempt
            .and_then(|index| self.attempts.get(index))
    }

    fn active_mut(&mut self) -> Result<&mut AttemptCheckpoint, CheckpointError> {
        let index = self
            .active_attempt
            .ok_or_else(|| CheckpointError::InvalidState("no active attempt".into()))?;
        self.attempts
            .get_mut(index)
            .ok_or_else(|| CheckpointError::InvalidState("active attempt index out of range".into()))
    }

    fn validate(&self) -> Result<(), CheckpointError> {
        if self.version != CHECKPOINT_VERSION {
            return Err(CheckpointError::Version(self.version));
        }
        if self.run_id.trim().is_empty() {
            return Err(CheckpointError::InvalidState("empty run id".into()));
        }
        if surface_digest(&self.accepted_surface) != self.accepted_surface_sha256 {
            return Err(CheckpointError::InvalidState(
                "accepted surface digest mismatch".into(),
            ));
        }
        if self.consumed != self.attempts.len() {
            return Err(CheckpointError::InvalidState(
                "consumed accounting does not match attempt history".into(),
            ));
        }
        let mut seen = BTreeSet::new();
        for (index, attempt) in self.attempts.iter().enumerate() {
            if attempt.index != index {
                return Err(CheckpointError::InvalidState(
                    "attempt index does not match history order".into(),
                ));
            }
            if !attempt.admitted || !attempt.charged {
                return Err(CheckpointError::InvalidState(
                    "attempt history contains an unadmitted or uncharged entry".into(),
                ));
            }
            if !seen.insert(attempt.attempt_id.clone()) {
                return Err(CheckpointError::InvalidState("duplicate attempt id".into()));
            }
            if let (Some(digest), Some(candidate)) =
                (&attempt.candidate_sha256, &attempt.candidate)
            {
                if surface_digest(candidate) != *digest {
                    return Err(CheckpointError::InvalidState(
                        "candidate digest mismatch".into(),
                    ));
                }
            }
            if let Some(evidence) = &attempt.evaluation {
                evidence.validate()?;
                if Some(&evidence.candidate_sha256) != attempt.candidate_sha256.as_ref() {
                    return Err(CheckpointError::InvalidState(
                        "evaluation evidence is bound to a different candidate".into(),
                    ));
                }
                if Some(&evidence.evaluator_config_sha256)
                    != attempt.evaluator_config_sha256.as_ref()
                {
                    return Err(CheckpointError::InvalidState(
                        "evaluation evidence is bound to a different evaluator".into(),
                    ));
                }
            }
            if matches!(
                attempt.phase,
                AttemptPhase::Evaluated | AttemptPhase::Promoted
            ) && attempt.evaluation.is_none()
            {
                return Err(CheckpointError::InvalidState(
                    "evaluated/promoted attempt has no evaluation evidence".into(),
                ));
            }
            if attempt.phase == AttemptPhase::Promoted
                && !attempt
                    .evaluation
                    .as_ref()
                    .is_some_and(|evidence| evidence.passed)
            {
                return Err(CheckpointError::InvalidState(
                    "promoted attempt has no passing evaluation".into(),
                ));
            }
            if attempt.phase == AttemptPhase::Abandoned && attempt.reconciliation.is_none() {
                return Err(CheckpointError::InvalidState(
                    "abandoned attempt has no reconciliation record".into(),
                ));
            }
            if attempt.provider != self.provider {
                return Err(CheckpointError::InvalidState(
                    "attempt provider binding does not match the run".into(),
                ));
            }
        }
        if let Some(index) = self.active_attempt {
            if index >= self.attempts.len() {
                return Err(CheckpointError::InvalidState(
                    "active attempt index out of range".into(),
                ));
            }
        }
        if self.status == RunStatus::Completed && self.active_attempt.is_some() {
            return Err(CheckpointError::InvalidState(
                "completed run still has an active attempt".into(),
            ));
        }
        if self.status == RunStatus::Blocked && self.block_reason.is_none() {
            return Err(CheckpointError::InvalidState(
                "blocked run has no persisted reason".into(),
            ));
        }
        if let Some(provider) = &self.provider {
            provider
                .validate()
                .map_err(|err| CheckpointError::InvalidState(err.to_string()))?;
        }
        Ok(())
    }

    pub fn has_incomplete_cleanup(&self) -> bool {
        self.cleanup.is_incomplete()
            || self.attempts.iter().any(|attempt| {
                attempt.reconciliation.is_none() && attempt.cleanup.is_incomplete()
            })
    }

    fn block_reason(&self) -> String {
        if let Some(attempt) = self.active() {
            format!(
                "attempt {} is {:?}",
                attempt.attempt_id, attempt.phase
            )
        } else if let Some(error) = self.cleanup.error() {
            format!("cleanup incomplete: {error}")
        } else {
            "run is blocked".to_string()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Recovery {
    /// No admitted/in-flight attempt; a new attempt may be admitted.
    Ready,
    /// The run completed; it must not execute again.
    Completed,
    /// In-flight/interrupted/cancelled/incomplete cleanup requires explicit reconciliation.
    Blocked(String),
    /// Cancelled with complete cleanup or terminally failed.
    Terminal(String),
}

struct OwnerLock {
    path: PathBuf,
    owner: String,
}

impl Drop for OwnerLock {
    fn drop(&mut self) {
        if let Ok(contents) = fs::read_to_string(&self.path) {
            if contents == self.owner {
                let _ = fs::remove_file(&self.path);
            }
        }
    }
}

pub struct CheckpointStore {
    dir: PathBuf,
    run_id: String,
    checkpoint: Checkpoint,
    _lock: OwnerLock,
    /// Set after a post-rename durability failure. No mutation is allowed until
    /// the on-disk state is loaded and adopted.
    uncertain: bool,
    #[cfg(test)]
    fault: Option<WriteFault>,
}

impl std::fmt::Debug for CheckpointStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointStore")
            .field("run_id", &self.run_id)
            .field("status", &self.checkpoint.status)
            .field("consumed", &self.checkpoint.consumed)
            .field("uncertain", &self.uncertain)
            .finish()
    }
}

impl CheckpointStore {
    pub fn create(
        dir: impl AsRef<Path>,
        run_id: &str,
        surface: &BTreeMap<PathBuf, String>,
    ) -> Result<Self, CheckpointError> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let lock = acquire_lock(&dir, run_id)?;
        let path = dir.join(CURRENT_FILE);
        if path.exists() {
            return Err(CheckpointError::AlreadyExists(path));
        }
        let checkpoint = Checkpoint::new(run_id, surface);
        let mut store = Self {
            dir,
            run_id: run_id.to_string(),
            checkpoint,
            _lock: lock,
            uncertain: false,
            #[cfg(test)]
            fault: None,
        };
        store.commit()?;
        Ok(store)
    }

    pub fn read_run_id(dir: impl AsRef<Path>) -> Result<String, CheckpointError> {
        let path = dir.as_ref().join(CURRENT_FILE);
        if !path.is_file() {
            return Err(CheckpointError::NotFound(path));
        }
        Ok(load_checkpoint(&path)?.run_id)
    }

    pub fn open(dir: impl AsRef<Path>, run_id: &str) -> Result<Self, CheckpointError> {
        let dir = dir.as_ref().to_path_buf();
        let path = dir.join(CURRENT_FILE);
        if !path.is_file() {
            return Err(CheckpointError::NotFound(path));
        }
        let lock = acquire_lock(&dir, run_id)?;
        let checkpoint = match load_checkpoint(&path) {
            Ok(checkpoint) => checkpoint,
            Err(err) => {
                drop(lock);
                return Err(err);
            }
        };
        if checkpoint.run_id != run_id {
            drop(lock);
            return Err(CheckpointError::Corrupt(format!(
                "run id mismatch: file has {:?}",
                checkpoint.run_id
            )));
        }
        checkpoint.validate()?;
        let mut store = Self {
            dir,
            run_id: run_id.to_string(),
            checkpoint,
            _lock: lock,
            uncertain: false,
            #[cfg(test)]
            fault: None,
        };
        // Opening a checkpoint must not expose a runnable store while an
        // admission/in-flight attempt or incomplete cleanup is outstanding.
        store.apply_recovery_barrier()?;
        Ok(store)
    }

    /// Bind the non-secret provider/model/config identity. Binding is
    /// immutable once recorded, and cannot be added to a run that already has
    /// admitted history.
    pub fn bind_provider(&mut self, binding: &ProviderBinding) -> Result<(), CheckpointError> {
        self.ensure_operable()?;
        binding
            .validate()
            .map_err(|err| CheckpointError::ConfigurationMismatch(err.to_string()))?;
        if let Some(existing) = &self.checkpoint.provider {
            if existing != binding {
                return Err(CheckpointError::ConfigurationMismatch(format!(
                    "checkpoint is bound to {}/{} but this request is {}/{}",
                    existing.provider,
                    existing.model,
                    binding.provider,
                    binding.model
                )));
            }
            return Ok(());
        }
        if !self.checkpoint.attempts.is_empty() {
            return Err(CheckpointError::ConfigurationMismatch(
                "checkpoint has attempt history but no provider binding".into(),
            ));
        }
        let binding = binding.clone();
        self.transact(|checkpoint| {
            checkpoint.provider = Some(binding);
            Ok(())
        })
    }

    pub fn checkpoint(&self) -> &Checkpoint {
        &self.checkpoint
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn is_uncertain(&self) -> bool {
        self.uncertain
    }

    /// Load the current on-disk state after an uncertain commit and adopt it.
    /// If the disk state is still invalid, the store remains blocked.
    pub fn reconcile_from_disk(&mut self) -> Result<Recovery, CheckpointError> {
        let checkpoint = load_checkpoint(&self.dir.join(CURRENT_FILE))?;
        if checkpoint.run_id != self.run_id {
            return Err(CheckpointError::Corrupt(format!(
                "run id mismatch: file has {:?}",
                checkpoint.run_id
            )));
        }
        checkpoint.validate()?;
        self.checkpoint = checkpoint;
        self.uncertain = false;
        self.apply_recovery_barrier()?;
        self.resume()
    }

    pub fn resume(&mut self) -> Result<Recovery, CheckpointError> {
        self.ensure_operable()?;
        if self.checkpoint.has_incomplete_cleanup()
            && !matches!(
                self.checkpoint.status,
                RunStatus::Blocked | RunStatus::CancelledCleanupIncomplete
            )
        {
            let reason = self
                .checkpoint
                .cleanup
                .error()
                .unwrap_or("cleanup incomplete")
                .to_string();
            self.mark_blocked(reason)?;
            return Ok(Recovery::Blocked(self.checkpoint.block_reason.clone().unwrap_or_default()));
        }
        if self.checkpoint.status == RunStatus::Running {
            if self.checkpoint.active_attempt.is_some() {
                let reason = self.checkpoint.block_reason();
                self.mark_blocked(reason)?;
                return Ok(Recovery::Blocked(self.checkpoint.block_reason.clone().unwrap_or_default()));
            }
            return Ok(Recovery::Ready);
        }
        match self.checkpoint.status {
            RunStatus::Completed => Ok(Recovery::Completed),
            RunStatus::Blocked => Ok(Recovery::Blocked(
                self.checkpoint
                    .block_reason
                    .clone()
                    .unwrap_or_else(|| self.checkpoint.block_reason()),
            )),
            RunStatus::CancelledCleanupIncomplete => {
                Ok(Recovery::Blocked("cancelled with incomplete cleanup".into()))
            }
            RunStatus::Cancelled => Ok(Recovery::Terminal("run was cancelled".into())),
            RunStatus::Exhausted => Ok(Recovery::Terminal("run budget was exhausted".into())),
            RunStatus::Failed => Ok(Recovery::Terminal("run failed".into())),
            RunStatus::Running => unreachable!(),
        }
    }

    fn mark_blocked(&mut self, reason: impl Into<String>) -> Result<(), CheckpointError> {
        let reason = reason.into();
        self.transact(|checkpoint| {
            checkpoint.status = RunStatus::Blocked;
            checkpoint.phase = Phase::Blocked;
            checkpoint.block_reason = Some(reason);
            Ok(())
        })
    }

    fn apply_recovery_barrier(&mut self) -> Result<(), CheckpointError> {
        let needs_block = self.checkpoint.has_incomplete_cleanup()
            || (self.checkpoint.status == RunStatus::Running
                && self.checkpoint.active_attempt.is_some());
        if needs_block
            && !matches!(
                self.checkpoint.status,
                RunStatus::Blocked | RunStatus::CancelledCleanupIncomplete
            )
        {
            let reason = if self.checkpoint.active_attempt.is_some() {
                self.checkpoint.block_reason()
            } else {
                self.checkpoint
                    .cleanup
                    .error()
                    .unwrap_or("cleanup incomplete")
                    .to_string()
            };
            self.mark_blocked(reason)?;
        }
        Ok(())
    }

    fn ensure_operable(&self) -> Result<(), CheckpointError> {
        if self.uncertain {
            return Err(CheckpointError::UncertainState(
                "on-disk state must be reconciled after an uncertain commit".into(),
            ));
        }
        Ok(())
    }

    fn transact<F>(&mut self, mutate: F) -> Result<(), CheckpointError>
    where
        F: FnOnce(&mut Checkpoint) -> Result<(), CheckpointError>,
    {
        self.ensure_operable()?;
        let mut next = self.checkpoint.clone();
        mutate(&mut next)?;
        next.validate()?;
        self.persist(&next)?;
        self.checkpoint = next;
        Ok(())
    }

    fn persist(&mut self, checkpoint: &Checkpoint) -> Result<(), CheckpointError> {
        let payload = serde_json::to_string(checkpoint)?;
        let envelope = Envelope {
            version: CHECKPOINT_VERSION,
            payload_sha256: sha256_hex(payload.as_bytes()),
            payload,
        };
        let bytes = serde_json::to_vec(&envelope)?;
        let path = self.dir.join(CURRENT_FILE);
        #[cfg(test)]
        let fault = self.fault.take().unwrap_or(WriteFault::None);
        #[cfg(not(test))]
        let fault = WriteFault::None;
        match atomic_write(&path, &bytes, fault) {
            Ok(()) => Ok(()),
            Err(failure) if failure.uncertain => {
                self.uncertain = true;
                Err(CheckpointError::Uncertain(failure.message))
            }
            Err(failure) => Err(CheckpointError::Uncommitted(failure.message)),
        }
    }

    pub fn admit_attempt(&mut self, attempt_id: &str) -> Result<usize, CheckpointError> {
        self.ensure_operable()?;
        if self.checkpoint.status != RunStatus::Running {
            return Err(CheckpointError::InvalidState(format!(
                "cannot admit in status {:?}",
                self.checkpoint.status
            )));
        }
        if self.checkpoint.block_reason.is_some() || self.checkpoint.has_incomplete_cleanup() {
            return Err(CheckpointError::InvalidState(
                "cannot admit while blocked or cleanup is unresolved".into(),
            ));
        }
        if self.checkpoint.active_attempt.is_some() {
            return Err(CheckpointError::InvalidState(
                "an attempt is already active".into(),
            ));
        }
        if self.checkpoint.attempts.iter().any(|attempt| attempt.attempt_id == attempt_id) {
            return Err(CheckpointError::InvalidState("duplicate attempt id".into()));
        }
        let index = self.checkpoint.attempts.len();
        self.transact(|checkpoint| {
            checkpoint.attempts.push(AttemptCheckpoint {
                attempt_id: attempt_id.to_string(),
                index,
                admitted: true,
                charged: true,
                phase: AttemptPhase::Admitted,
                candidate_sha256: None,
                candidate: None,
                evaluation: None,
                evaluator_config_sha256: None,
                failure: None,
                cleanup: CleanupState::NotRun,
                partial: None,
                provider: checkpoint.provider.clone(),
                usage: None,
                reconciliation: None,
            });
            checkpoint.consumed += 1;
            checkpoint.active_attempt = Some(index);
            checkpoint.phase = Phase::Admitted;
            checkpoint.block_reason = None;
            Ok(())
        })?;
        Ok(index)
    }

    pub fn begin_generation(&mut self) -> Result<(), CheckpointError> {
        self.transact(|checkpoint| {
            require_running(checkpoint)?;
            let attempt = checkpoint.active_mut()?;
            require_phase(attempt, AttemptPhase::Admitted)?;
            attempt.phase = AttemptPhase::Generating;
            checkpoint.phase = Phase::Generation;
            Ok(())
        })
    }

    pub fn begin_probe(&mut self) -> Result<(), CheckpointError> {
        self.transact(|checkpoint| {
            require_running(checkpoint)?;
            let attempt = checkpoint.active_mut()?;
            require_phase(attempt, AttemptPhase::Generating)?;
            attempt.phase = AttemptPhase::ProbeInFlight;
            checkpoint.phase = Phase::Probe;
            Ok(())
        })
    }

    pub fn probe_complete(
        &mut self,
        candidate: &BTreeMap<PathBuf, String>,
    ) -> Result<(), CheckpointError> {
        let candidate = surface_map(candidate);
        let digest = surface_digest(&candidate);
        self.transact(|checkpoint| {
            require_running(checkpoint)?;
            let attempt = checkpoint.active_mut()?;
            require_phase(attempt, AttemptPhase::ProbeInFlight)?;
            attempt.candidate = Some(candidate);
            attempt.candidate_sha256 = Some(digest);
            attempt.phase = AttemptPhase::ProbeComplete;
            checkpoint.phase = Phase::Evaluation;
            Ok(())
        })
    }

    /// Unchanged surfaces are verified without a probe. Bind the unchanged
    /// candidate and evaluator configuration, then enter evaluation.
    pub fn begin_unchanged_verification(
        &mut self,
        candidate: &BTreeMap<PathBuf, String>,
        evaluator_config_sha256: &str,
    ) -> Result<(), CheckpointError> {
        if evaluator_config_sha256.trim().is_empty() {
            return Err(CheckpointError::InvalidState(
                "evaluator configuration digest is required".into(),
            ));
        }
        let candidate = surface_map(candidate);
        let digest = surface_digest(&candidate);
        let evaluator_config_sha256 = evaluator_config_sha256.to_string();
        self.transact(|checkpoint| {
            require_running(checkpoint)?;
            let attempt = checkpoint.active_mut()?;
            require_phase(attempt, AttemptPhase::Generating)?;
            attempt.candidate = Some(candidate);
            attempt.candidate_sha256 = Some(digest);
            attempt.evaluator_config_sha256 = Some(evaluator_config_sha256);
            attempt.phase = AttemptPhase::EvaluationInFlight;
            checkpoint.phase = Phase::Evaluation;
            Ok(())
        })
    }

    pub fn begin_evaluation(
        &mut self,
        evaluator_config_sha256: &str,
    ) -> Result<(), CheckpointError> {
        if evaluator_config_sha256.trim().is_empty() {
            return Err(CheckpointError::InvalidState(
                "evaluator configuration digest is required".into(),
            ));
        }
        let evaluator_config_sha256 = evaluator_config_sha256.to_string();
        self.transact(|checkpoint| {
            require_running(checkpoint)?;
            let attempt = checkpoint.active_mut()?;
            require_phase(attempt, AttemptPhase::ProbeComplete)?;
            attempt.evaluator_config_sha256 = Some(evaluator_config_sha256);
            attempt.phase = AttemptPhase::EvaluationInFlight;
            checkpoint.phase = Phase::Evaluation;
            Ok(())
        })
    }

    pub fn evaluation_complete(
        &mut self,
        evidence: EvaluationEvidence,
    ) -> Result<(), CheckpointError> {
        evidence.validate()?;
        self.transact(|checkpoint| {
            require_running(checkpoint)?;
            let attempt = checkpoint.active_mut()?;
            require_phase(attempt, AttemptPhase::EvaluationInFlight)?;
            if Some(&evidence.candidate_sha256) != attempt.candidate_sha256.as_ref() {
                return Err(CheckpointError::InvalidState(
                    "evaluation evidence is bound to a different candidate".into(),
                ));
            }
            if Some(&evidence.evaluator_config_sha256)
                != attempt.evaluator_config_sha256.as_ref()
            {
                return Err(CheckpointError::InvalidState(
                    "evaluation evidence is bound to a different evaluator".into(),
                ));
            }
            attempt.evaluation = Some(evidence);
            attempt.phase = AttemptPhase::Evaluated;
            checkpoint.phase = Phase::Promotion;
            Ok(())
        })
    }

    pub fn promote(&mut self) -> Result<(), CheckpointError> {
        self.transact(|checkpoint| {
            require_running(checkpoint)?;
            if checkpoint.block_reason.is_some() || checkpoint.has_incomplete_cleanup() {
                return Err(CheckpointError::InvalidState(
                    "cannot promote a blocked or cleanup-incomplete run".into(),
                ));
            }
            let index = checkpoint
                .active_attempt
                .ok_or_else(|| CheckpointError::InvalidState("no active attempt".into()))?;
            let attempt = &checkpoint.attempts[index];
            require_phase(attempt, AttemptPhase::Evaluated)?;
            let evidence = attempt
                .evaluation
                .as_ref()
                .ok_or_else(|| CheckpointError::InvalidState("attempt has no evaluation".into()))?;
            evidence.validate()?;
            if !evidence.passed {
                return Err(CheckpointError::InvalidState(
                    "refusing to promote an unevaluated or failing candidate".into(),
                ));
            }
            let candidate = attempt
                .candidate
                .clone()
                .ok_or_else(|| CheckpointError::InvalidState("attempt has no candidate".into()))?;
            if surface_digest(&candidate) != attempt.candidate_sha256.clone().unwrap_or_default() {
                return Err(CheckpointError::InvalidState(
                    "candidate digest mismatch".into(),
                ));
            }
            if Some(&evidence.candidate_sha256) != attempt.candidate_sha256.as_ref()
                || Some(&evidence.evaluator_config_sha256)
                    != attempt.evaluator_config_sha256.as_ref()
            {
                return Err(CheckpointError::InvalidState(
                    "evaluation evidence binding no longer matches the candidate/config".into(),
                ));
            }
            checkpoint.accepted_surface = candidate;
            checkpoint.accepted_surface_sha256 = surface_digest(&checkpoint.accepted_surface);
            checkpoint.attempts[index].phase = AttemptPhase::Promoted;
            checkpoint.active_attempt = None;
            checkpoint.status = RunStatus::Completed;
            checkpoint.phase = Phase::Terminal;
            checkpoint.block_reason = None;
            Ok(())
        })
    }

    pub fn fail_attempt(&mut self, reason: impl Into<String>) -> Result<(), CheckpointError> {
        let reason = reason.into();
        self.transact(|checkpoint| {
            require_running(checkpoint)?;
            if checkpoint.block_reason.is_some() || checkpoint.has_incomplete_cleanup() {
                return Err(CheckpointError::InvalidState(
                    "cannot fail an attempt while blocked or cleanup is unresolved".into(),
                ));
            }
            let index = checkpoint
                .active_attempt
                .ok_or_else(|| CheckpointError::InvalidState("no active attempt".into()))?;
            match checkpoint.attempts[index].phase {
                AttemptPhase::Promoted
                | AttemptPhase::Rejected
                | AttemptPhase::Cancelled
                | AttemptPhase::Abandoned => {
                    return Err(CheckpointError::InvalidState(
                        "attempt is already terminal".into(),
                    ));
                }
                _ => {}
            }
            checkpoint.attempts[index].phase = AttemptPhase::Rejected;
            checkpoint.attempts[index].failure = Some(reason);
            checkpoint.active_attempt = None;
            checkpoint.status = RunStatus::Running;
            checkpoint.phase = Phase::Idle;
            checkpoint.block_reason = None;
            Ok(())
        })
    }

    pub fn cancel(&mut self, cleanup: CleanupState) -> Result<(), CheckpointError> {
        self.ensure_operable()?;
        if matches!(
            self.checkpoint.status,
            RunStatus::Completed
                | RunStatus::Failed
                | RunStatus::Cancelled
                | RunStatus::CancelledCleanupIncomplete
                | RunStatus::Exhausted
        ) {
            return Err(CheckpointError::InvalidState(
                "cannot cancel a terminal run".into(),
            ));
        }
        let existing_cleanup = self
            .checkpoint
            .cleanup
            .is_incomplete()
            .then(|| self.checkpoint.cleanup.clone())
            .or_else(|| {
                self.checkpoint
                    .attempts
                    .iter()
                    .find(|attempt| {
                        attempt.reconciliation.is_none() && attempt.cleanup.is_incomplete()
                    })
                    .map(|attempt| attempt.cleanup.clone())
            });
        if existing_cleanup.is_some() && !cleanup.is_incomplete() {
            return Err(CheckpointError::InvalidState(
                "cannot downgrade unresolved cleanup during cancellation".into(),
            ));
        }
        let effective_cleanup = existing_cleanup.unwrap_or(cleanup);
        self.transact(|checkpoint| {
            checkpoint.cancellation = CancellationState::Observed;
            checkpoint.cleanup = effective_cleanup.clone();
            if let Some(index) = checkpoint.active_attempt {
                checkpoint.attempts[index].phase = AttemptPhase::Cancelled;
                checkpoint.attempts[index].cleanup = effective_cleanup.clone();
                checkpoint.attempts[index].reconciliation = None;
            }
            checkpoint.status = if effective_cleanup.is_incomplete() {
                RunStatus::CancelledCleanupIncomplete
            } else {
                RunStatus::Cancelled
            };
            checkpoint.phase = Phase::Terminal;
            checkpoint.block_reason = effective_cleanup.error().map(|error| {
                format!("cancelled with incomplete cleanup: {error}")
            });
            Ok(())
        })
    }

    pub fn record_usage(&mut self, usage: Usage) -> Result<(), CheckpointError> {
        self.transact(|checkpoint| {
            let index = checkpoint
                .active_attempt
                .ok_or_else(|| CheckpointError::InvalidState("no active attempt".into()))?;
            checkpoint.attempts[index].usage = Some(usage);
            Ok(())
        })
    }

    pub fn record_cleanup(
        &mut self,
        cleanup: CleanupState,
        partial: Option<&PartialOutput>,
    ) -> Result<(), CheckpointError> {
        self.ensure_operable()?;
        if matches!(
            self.checkpoint.status,
            RunStatus::Completed
                | RunStatus::Failed
                | RunStatus::Cancelled
                | RunStatus::Exhausted
        ) {
            return Err(CheckpointError::InvalidState(
                "cannot record cleanup on a terminal run".into(),
            ));
        }
        if self.checkpoint.has_incomplete_cleanup() {
            if !cleanup.is_incomplete() {
                return Err(CheckpointError::InvalidState(
                    "cannot clear unresolved cleanup".into(),
                ));
            }
            // Keep the original unresolved error and partial evidence. If the
            // current attempt has no diagnostics yet, add them without
            // replacing the barrier or prior evidence.
            let active_missing_partial = self
                .checkpoint
                .active_attempt
                .and_then(|index| self.checkpoint.attempts.get(index))
                .is_some_and(|attempt| attempt.partial.is_none());
            if !active_missing_partial || partial.is_none() {
                return Ok(());
            }
            return self.transact(|checkpoint| {
                if let Some(index) = checkpoint.active_attempt {
                    if checkpoint.attempts[index].partial.is_none() {
                        checkpoint.attempts[index].partial =
                            partial.map(PartialSnapshot::from);
                    }
                }
                Ok(())
            });
        }

        self.transact(|checkpoint| {
            checkpoint.cleanup = cleanup.clone();
            if let Some(index) = checkpoint.active_attempt {
                checkpoint.attempts[index].cleanup = cleanup.clone();
                // A no-diagnostic update must not erase useful existing
                // diagnostics for this attempt.
                if let Some(partial) = partial {
                    checkpoint.attempts[index].partial = Some(PartialSnapshot::from(partial));
                }
            }
            if cleanup.is_incomplete() {
                checkpoint.status = RunStatus::Blocked;
                checkpoint.phase = Phase::Blocked;
                checkpoint.block_reason = Some(format!(
                    "cleanup incomplete: {}",
                    cleanup.error().unwrap_or("unknown")
                ));
            }
            Ok(())
        })
    }

    pub fn block(&mut self, reason: impl Into<String>) -> Result<(), CheckpointError> {
        self.ensure_operable()?;
        if matches!(
            self.checkpoint.status,
            RunStatus::Completed
                | RunStatus::Failed
                | RunStatus::Cancelled
                | RunStatus::CancelledCleanupIncomplete
                | RunStatus::Exhausted
        ) {
            return Err(CheckpointError::InvalidState(
                "cannot block a terminal run".into(),
            ));
        }
        self.mark_blocked(reason)
    }

    /// Explicit trusted reconciliation. A reason string alone is never enough;
    /// the caller must affirm that execution uncertainty is resolved and cleanup
    /// was verified. Historical failure and cleanup evidence are preserved.
    pub fn reconcile_abandon(
        &mut self,
        confirmation: ReconciliationConfirmation,
    ) -> Result<(), CheckpointError> {
        self.ensure_operable()?;
        if confirmation.operator.trim().is_empty()
            || confirmation.note.trim().is_empty()
            || !confirmation.execution_uncertainty_resolved
            || !confirmation.cleanup_verified
        {
            return Err(CheckpointError::InvalidState(
                "reconciliation requires trusted operator confirmation and verified cleanup".into(),
            ));
        }
        let can_reconcile = matches!(
            self.checkpoint.status,
            RunStatus::Blocked | RunStatus::CancelledCleanupIncomplete
        ) || self.checkpoint.has_incomplete_cleanup();
        if !can_reconcile {
            return Err(CheckpointError::InvalidState(
                "run is not blocked or cleanup-incomplete".into(),
            ));
        }
        let resolved_attempt_id = self
            .checkpoint
            .active()
            .map(|attempt| attempt.attempt_id.clone());
        let reconciliation = Reconciliation {
            operator: confirmation.operator,
            note: confirmation.note,
            execution_uncertainty_resolved: true,
            cleanup_verified: true,
            resolved_attempt_id: resolved_attempt_id.clone(),
        };
        self.transact(|checkpoint| {
            if let Some(index) = checkpoint.active_attempt.take() {
                checkpoint.attempts[index].phase = AttemptPhase::Abandoned;
                checkpoint.attempts[index].reconciliation = Some(reconciliation.clone());
            }
            checkpoint.reconciliation = Some(reconciliation);
            checkpoint.status = RunStatus::Running;
            checkpoint.phase = Phase::Idle;
            checkpoint.cleanup = CleanupState::NotRun;
            checkpoint.block_reason = None;
            checkpoint.cancellation = CancellationState::NotRequested;
            Ok(())
        })
    }

    pub fn finish_exhausted(&mut self) -> Result<(), CheckpointError> {
        self.transact(|checkpoint| {
            if checkpoint.active_attempt.is_some() {
                return Err(CheckpointError::InvalidState(
                    "cannot exhaust a run with an active attempt".into(),
                ));
            }
            if !matches!(checkpoint.status, RunStatus::Running | RunStatus::Blocked) {
                return Err(CheckpointError::InvalidState(
                    "cannot exhaust a terminal run".into(),
                ));
            }
            if checkpoint.has_incomplete_cleanup() {
                return Err(CheckpointError::InvalidState(
                    "cannot exhaust a run with unresolved cleanup".into(),
                ));
            }
            checkpoint.status = RunStatus::Exhausted;
            checkpoint.phase = Phase::Terminal;
            checkpoint.block_reason = None;
            Ok(())
        })
    }

    pub fn finish_failed(&mut self, reason: impl Into<String>) -> Result<(), CheckpointError> {
        let reason = reason.into();
        self.transact(|checkpoint| {
            if matches!(
                checkpoint.status,
                RunStatus::Completed
                    | RunStatus::Exhausted
                    | RunStatus::Cancelled
                    | RunStatus::CancelledCleanupIncomplete
            ) {
                return Err(CheckpointError::InvalidState(
                    "cannot fail a terminal run".into(),
                ));
            }
            if checkpoint.has_incomplete_cleanup() {
                return Err(CheckpointError::InvalidState(
                    "cannot fail a run with unresolved cleanup".into(),
                ));
            }
            if let Some(index) = checkpoint.active_attempt.take() {
                checkpoint.attempts[index].phase = AttemptPhase::Rejected;
                checkpoint.attempts[index].failure = Some(reason.clone());
            }
            checkpoint.status = RunStatus::Failed;
            checkpoint.phase = Phase::Terminal;
            checkpoint.block_reason = Some(reason);
            Ok(())
        })
    }

    pub fn commit(&mut self) -> Result<(), CheckpointError> {
        self.ensure_operable()?;
        self.checkpoint.validate()?;
        let checkpoint = self.checkpoint.clone();
        self.persist(&checkpoint)
    }

    #[cfg(test)]
    fn inject_fault(&mut self, fault: WriteFault) {
        self.fault = Some(fault);
    }
}

#[derive(Serialize, Deserialize)]
struct Envelope {
    version: u32,
    payload_sha256: String,
    payload: String,
}

fn load_checkpoint(path: &Path) -> Result<Checkpoint, CheckpointError> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let envelope: Envelope = serde_json::from_slice(&bytes)
        .map_err(|err| CheckpointError::Corrupt(format!("envelope: {err}")))?;
    if envelope.version != CHECKPOINT_VERSION {
        return Err(CheckpointError::Version(envelope.version));
    }
    if sha256_hex(envelope.payload.as_bytes()) != envelope.payload_sha256 {
        return Err(CheckpointError::Corrupt(
            "payload checksum mismatch".into(),
        ));
    }
    let checkpoint: Checkpoint = serde_json::from_str(&envelope.payload)
        .map_err(|err| CheckpointError::Corrupt(format!("payload: {err}")))?;
    checkpoint.validate()?;
    Ok(checkpoint)
}

fn acquire_lock(dir: &Path, run_id: &str) -> Result<OwnerLock, CheckpointError> {
    let path = dir.join(LOCK_FILE);
    let owner = format!(
        "{}-{}-{}-{}",
        run_id,
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default(),
        OWNER_COUNTER.fetch_add(1, Ordering::SeqCst)
    );
    match OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(mut file) => {
            file.write_all(owner.as_bytes())?;
            file.sync_all()?;
            Ok(OwnerLock { path, owner })
        }
        Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {
            Err(CheckpointError::Locked(path))
        }
        Err(err) => Err(CheckpointError::Io(err)),
    }
}

struct WriteFailure {
    message: String,
    uncertain: bool,
}

impl WriteFailure {
    fn definite(context: &str, err: io::Error) -> Self {
        Self {
            message: format!("{context}: {err}"),
            uncertain: false,
        }
    }

    fn uncertain(context: &str, err: io::Error) -> Self {
        Self {
            message: format!("{context}: {err}"),
            uncertain: true,
        }
    }
}

fn atomic_write(path: &Path, bytes: &[u8], fault: WriteFault) -> Result<(), WriteFailure> {
    let dir = path.parent().ok_or_else(|| WriteFailure {
        message: "checkpoint path has no parent".into(),
        uncertain: false,
    })?;
    let temp = dir.join(format!(
        "{CURRENT_FILE}.tmp.{}.{}",
        std::process::id(),
        OWNER_COUNTER.fetch_add(1, Ordering::SeqCst)
    ));
    let result = (|| -> Result<(), WriteFailure> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|err| WriteFailure::definite("create temporary checkpoint", err))?;
        let write_bytes = match fault {
            WriteFault::PartialWrite => &bytes[..bytes.len() / 2],
            _ => bytes,
        };
        file.write_all(write_bytes)
            .map_err(|err| WriteFailure::definite("write temporary checkpoint", err))?;
        file.sync_all()
            .map_err(|err| WriteFailure::definite("sync temporary checkpoint", err))?;
        drop(file);
        if fault == WriteFault::BeforeRename {
            return Err(WriteFailure {
                message: "injected failure before rename".into(),
                uncertain: false,
            });
        }
        if fault == WriteFault::PartialWrite {
            return Err(WriteFailure {
                message: "injected partial write".into(),
                uncertain: false,
            });
        }
        fs::rename(&temp, path)
            .map_err(|err| WriteFailure::definite("rename checkpoint", err))?;
        let dir_file = File::open(dir)
            .map_err(|err| WriteFailure::uncertain("open checkpoint directory for sync", err))?;
        dir_file
            .sync_all()
            .map_err(|err| WriteFailure::uncertain("sync checkpoint directory", err))?;
        if fault == WriteFault::AfterRename {
            return Err(WriteFailure {
                message: "injected failure after rename".into(),
                uncertain: true,
            });
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WriteFault {
    None,
    BeforeRename,
    AfterRename,
    PartialWrite,
}

fn require_running(checkpoint: &Checkpoint) -> Result<(), CheckpointError> {
    if checkpoint.status != RunStatus::Running {
        return Err(CheckpointError::InvalidState(format!(
            "run is {:?}, not running",
            checkpoint.status
        )));
    }
    if checkpoint.block_reason.is_some() || checkpoint.has_incomplete_cleanup() {
        return Err(CheckpointError::InvalidState(
            "run is blocked or cleanup is unresolved".into(),
        ));
    }
    Ok(())
}

fn require_phase(
    attempt: &AttemptCheckpoint,
    expected: AttemptPhase,
) -> Result<(), CheckpointError> {
    if attempt.phase != expected {
        return Err(CheckpointError::InvalidState(format!(
            "expected attempt phase {expected:?}, found {:?}",
            attempt.phase
        )));
    }
    Ok(())
}

fn surface_map(surface: &BTreeMap<PathBuf, String>) -> BTreeMap<String, String> {
    surface
        .iter()
        .map(|(path, content)| (path.to_string_lossy().into_owned(), content.clone()))
        .collect()
}

pub fn tree_digest(surface: &BTreeMap<PathBuf, String>) -> String {
    surface_digest(&surface_map(surface))
}

pub fn surface_digest(surface: &BTreeMap<String, String>) -> String {
    let mut bytes = Vec::new();
    for (path, content) in surface {
        bytes.extend_from_slice(&(path.len() as u64).to_le_bytes());
        bytes.extend_from_slice(path.as_bytes());
        bytes.extend_from_slice(&(content.len() as u64).to_le_bytes());
        bytes.extend_from_slice(content.as_bytes());
    }
    sha256_hex(&bytes)
}

pub fn new_run_id() -> String {
    format!(
        "run-{}-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default(),
        std::process::id()
    )
}

// ---------------------------------------------------------------------------
// SHA-256 (checkpoint integrity and surface identity; not a secret store)
// ---------------------------------------------------------------------------

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

pub fn sha256_hex(input: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
        0x1f83d9ab, 0x5be0cd19,
    ];
    let mut msg = input.to_vec();
    let bit_len = (input.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (index, word) in chunk.chunks_exact(4).enumerate() {
            w[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }
        let mut a = h;
        for index in 0..64 {
            let s1 = a[4].rotate_right(6) ^ a[4].rotate_right(11) ^ a[4].rotate_right(25);
            let ch = (a[4] & a[5]) ^ ((!a[4]) & a[6]);
            let temp1 = a[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[index])
                .wrapping_add(w[index]);
            let s0 = a[0].rotate_right(2) ^ a[0].rotate_right(13) ^ a[0].rotate_right(22);
            let maj = (a[0] & a[1]) ^ (a[0] & a[2]) ^ (a[1] & a[2]);
            let temp2 = s0.wrapping_add(maj);
            a = [
                temp1.wrapping_add(temp2),
                a[0],
                a[1],
                a[2],
                a[3].wrapping_add(temp1),
                a[4],
                a[5],
                a[6],
            ];
        }
        for index in 0..8 {
            h[index] = h[index].wrapping_add(a[index]);
        }
    }
    h.iter().map(|word| format!("{word:08x}")).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn tree(entries: &[(&str, &str)]) -> BTreeMap<PathBuf, String> {
        entries
            .iter()
            .map(|(path, content)| (PathBuf::from(path), (*content).to_string()))
            .collect()
    }

    const EVAL_CONFIG: &str = "eval-config-1";

    fn confirmation() -> ReconciliationConfirmation {
        ReconciliationConfirmation {
            operator: "test-operator".into(),
            note: "verified no process or execution remains".into(),
            execution_uncertainty_resolved: true,
            cleanup_verified: true,
        }
    }

    fn evidence(store: &CheckpointStore) -> EvaluationEvidence {
        let attempt = store.checkpoint().active().expect("active");
        EvaluationEvidence {
            passed: true,
            candidate_sha256: attempt.candidate_sha256.clone().expect("candidate"),
            evaluator_config_sha256: EVAL_CONFIG.into(),
            case_count: 1,
            failure_count: 0,
            cases: vec![CaseEvidence {
                name: "case".into(),
                passed: true,
                exit_code: 0,
                actual_sha256: "actual".into(),
                stderr_sha256: "stderr".into(),
            }],
        }
    }

    fn admitted_store(dir: &TempDir) -> CheckpointStore {
        let mut store = CheckpointStore::create(dir.path(), "run-1", &tree(&[("a", "1")]))
            .expect("create");
        store.admit_attempt("attempt-1").expect("admit");
        store
    }

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn admission_is_committed_before_external_work() {
        let dir = TempDir::new().expect("dir");
        let mut store = admitted_store(&dir);
        store.begin_generation().expect("generation");
        store.begin_probe().expect("probe");
        assert_eq!(store.checkpoint().consumed, 1);
        assert_eq!(
            store.checkpoint().active().expect("active").phase,
            AttemptPhase::ProbeInFlight
        );
    }

    #[test]
    fn interrupted_attempt_blocks_and_does_not_recharge() {
        let dir = TempDir::new().expect("dir");
        {
            let mut store = admitted_store(&dir);
            store.begin_generation().expect("gen");
            store.begin_probe().expect("probe");
        }
        let mut store = CheckpointStore::open(dir.path(), "run-1").expect("open");
        assert_eq!(store.checkpoint().status, RunStatus::Blocked);
        assert!(matches!(store.resume().expect("resume"), Recovery::Blocked(_)));
        assert_eq!(store.checkpoint().consumed, 1);
        assert_eq!(store.checkpoint().attempts.len(), 1);
        assert!(store.admit_attempt("attempt-2").is_err());
    }

    #[test]
    fn duplicate_recovery_owner_is_rejected() {
        let dir = TempDir::new().expect("dir");
        let store = admitted_store(&dir);
        let second = CheckpointStore::open(dir.path(), "run-1");
        assert!(matches!(second, Err(CheckpointError::Locked(_))));
        drop(store);
        let third = CheckpointStore::open(dir.path(), "run-1").expect("after release");
        assert_eq!(third.checkpoint().consumed, 1);
    }

    #[test]
    fn unsupported_checkpoint_version_fails_closed() {
        let dir = TempDir::new().expect("dir");
        {
            let _store = admitted_store(&dir);
        }
        let path = dir.path().join(CURRENT_FILE);
        let text = fs::read_to_string(&path).expect("read");
        fs::write(&path, text.replacen("\"version\":3", "\"version\":4", 1)).expect("write");
        assert!(matches!(
            CheckpointStore::open(dir.path(), "run-1"),
            Err(CheckpointError::Version(4))
        ));
    }

    #[test]
    fn corrupt_and_truncated_checkpoints_fail_closed() {
        let dir = TempDir::new().expect("dir");
        {
            let _store = admitted_store(&dir);
        }
        let path = dir.path().join(CURRENT_FILE);
        fs::write(&path, b"not-json").expect("write");
        assert!(matches!(
            CheckpointStore::open(dir.path(), "run-1"),
            Err(CheckpointError::Corrupt(_))
        ));
        fs::write(&path, b"{\"version\":1").expect("truncate");
        assert!(matches!(
            CheckpointStore::open(dir.path(), "run-1"),
            Err(CheckpointError::Corrupt(_))
        ));
        fs::remove_file(&path).expect("remove");
        assert!(matches!(
            CheckpointStore::open(dir.path(), "run-1"),
            Err(CheckpointError::NotFound(_))
        ));
    }

    #[test]
    fn admission_write_failure_does_not_leave_a_committed_attempt() {
        let dir = TempDir::new().expect("dir");
        {
            let mut store = CheckpointStore::create(
                dir.path(),
                "run-1",
                &tree(&[("a", "1")]),
            )
            .expect("create");
            store.inject_fault(WriteFault::BeforeRename);
            assert!(matches!(
                store.admit_attempt("attempt-1"),
                Err(CheckpointError::Uncommitted(_))
            ));
            assert_eq!(store.checkpoint().consumed, 0);
            assert!(store.checkpoint().attempts.is_empty());
            // Continue using the same store after a definitely-uncommitted write.
            store.admit_attempt("attempt-1").expect("retry admission");
            assert_eq!(store.checkpoint().consumed, 1);
        }
        let mut store = CheckpointStore::open(dir.path(), "run-1").expect("open");
        assert_eq!(store.checkpoint().consumed, 1);
        assert!(matches!(store.resume().expect("resume"), Recovery::Blocked(_)));
    }

    #[test]
    fn evaluation_in_flight_blocks_and_preserves_history() {
        let dir = TempDir::new().expect("dir");
        {
            let mut store = admitted_store(&dir);
            let candidate = tree(&[("a", "2")]);
            store.begin_generation().expect("gen");
            store.begin_probe().expect("probe");
            store.probe_complete(&candidate).expect("probe done");
            store.begin_evaluation(EVAL_CONFIG).expect("eval");
        }
        let mut store = CheckpointStore::open(dir.path(), "run-1").expect("open");
        assert!(matches!(store.resume().expect("resume"), Recovery::Blocked(_)));
        assert_eq!(store.checkpoint().consumed, 1);
        assert_eq!(
            store.checkpoint().active().expect("active").phase,
            AttemptPhase::EvaluationInFlight
        );
        assert_eq!(store.checkpoint().active().expect("active").attempt_id, "attempt-1");
    }

    #[test]
    fn promotion_write_failure_keeps_previous_surface() {
        let dir = TempDir::new().expect("dir");
        {
            let mut store = admitted_store(&dir);
            let candidate = tree(&[("a", "2")]);
            store.begin_generation().expect("gen");
            store.begin_probe().expect("probe");
            store.probe_complete(&candidate).expect("probe done");
            store.begin_evaluation(EVAL_CONFIG).expect("eval");
            let evidence = evidence(&store);
            store.evaluation_complete(evidence).expect("evidence");
            store.inject_fault(WriteFault::BeforeRename);
            assert!(store.promote().is_err());
        }
        let mut store = CheckpointStore::open(dir.path(), "run-1").expect("open");
        assert_eq!(store.checkpoint().status, RunStatus::Blocked);
        assert_eq!(store.checkpoint().accepted_surface.get("a"), Some(&"1".to_string()));
        assert!(matches!(store.resume().expect("resume"), Recovery::Blocked(_)));
    }

    #[test]
    fn tampered_surface_or_candidate_digest_is_rejected() {
        let dir = TempDir::new().expect("dir");
        let mut store = admitted_store(&dir);
        store
            .checkpoint
            .accepted_surface
            .insert("evil".into(), "injected".into());
        assert!(store.commit().is_err());
        store.checkpoint.accepted_surface.remove("evil");
        store.commit().expect("restore committed surface");

        let candidate = tree(&[("a", "2")]);
        store.begin_generation().expect("gen");
        store.begin_probe().expect("probe");
        store.probe_complete(&candidate).expect("probe done");
        store
            .checkpoint
            .attempts[0]
            .candidate
            .as_mut()
            .expect("candidate")
            .insert("a".into(), "substituted".into());
        assert!(store.commit().is_err());
    }

    #[test]
    fn incomplete_cleanup_on_a_running_attempt_blocks_recovery() {
        let dir = TempDir::new().expect("dir");
        let partial = PartialOutput {
            stdout: "partial".into(),
            cleanup_incomplete: true,
            cleanup_error: Some("reap stuck".into()),
            ..PartialOutput::default()
        };
        {
            let mut store = admitted_store(&dir);
            store.begin_generation().expect("gen");
            store.begin_probe().expect("probe");
            store
                .record_cleanup(
                    CleanupState::Incomplete {
                        error: "reap stuck".into(),
                    },
                    Some(&partial),
                )
                .expect("cleanup");
        }
        let mut store = CheckpointStore::open(dir.path(), "run-1").expect("open");
        assert!(matches!(store.resume().expect("resume"), Recovery::Blocked(_)));
        assert_eq!(store.checkpoint().consumed, 1);
        assert!(store.checkpoint().cleanup.is_incomplete());
        assert!(store.checkpoint().attempts[0].partial.is_some());
        store.reconcile_abandon(confirmation()).expect("reconcile");
        assert_eq!(store.checkpoint().status, RunStatus::Running);
    }

    #[test]
    fn cancellation_state_and_complete_cleanup_are_terminal() {
        let dir = TempDir::new().expect("dir");
        {
            let mut store = admitted_store(&dir);
            store.cancel(CleanupState::Complete).expect("cancel");
            assert_eq!(store.checkpoint().cancellation, CancellationState::Observed);
        }
        let mut store = CheckpointStore::open(dir.path(), "run-1").expect("open");
        assert!(matches!(store.resume().expect("resume"), Recovery::Terminal(_)));
        assert!(store.admit_attempt("attempt-2").is_err());
    }

    #[test]
    fn interrupted_write_keeps_last_committed_state() {
        let dir = TempDir::new().expect("dir");
        {
            let mut store = admitted_store(&dir);
            store.inject_fault(WriteFault::BeforeRename);
            assert!(store.begin_generation().is_err());
        }
        let mut store = CheckpointStore::open(dir.path(), "run-1").expect("open");
        assert_eq!(
            store.checkpoint().active().expect("active").phase,
            AttemptPhase::Admitted
        );
        assert_eq!(store.checkpoint().consumed, 1);
        assert!(matches!(store.resume().expect("resume"), Recovery::Blocked(_)));
    }

    #[test]
    fn partial_temp_write_keeps_last_committed_state() {
        let dir = TempDir::new().expect("dir");
        {
            let mut store = admitted_store(&dir);
            store.inject_fault(WriteFault::PartialWrite);
            assert!(store.begin_generation().is_err());
        }
        let mut store = CheckpointStore::open(dir.path(), "run-1").expect("open");
        assert_eq!(
            store.checkpoint().active().expect("active").phase,
            AttemptPhase::Admitted
        );
        assert!(matches!(store.resume().expect("resume"), Recovery::Blocked(_)));
    }

    #[test]
    fn failure_after_rename_commits_new_state() {
        let dir = TempDir::new().expect("dir");
        {
            let mut store = admitted_store(&dir);
            store.inject_fault(WriteFault::AfterRename);
            assert!(store.begin_generation().is_err());
        }
        let mut store = CheckpointStore::open(dir.path(), "run-1").expect("open");
        assert_eq!(
            store.checkpoint().active().expect("active").phase,
            AttemptPhase::Generating
        );
        assert!(matches!(store.resume().expect("resume"), Recovery::Blocked(_)));
    }

    #[test]
    fn failed_mutation_does_not_change_memory_and_store_remains_usable() {
        let dir = TempDir::new().expect("dir");
        let mut store = admitted_store(&dir);
        store.inject_fault(WriteFault::BeforeRename);
        assert!(matches!(
            store.begin_generation(),
            Err(CheckpointError::Uncommitted(_))
        ));
        assert_eq!(
            store.checkpoint().active().expect("active").phase,
            AttemptPhase::Admitted
        );
        store.begin_generation().expect("retry");
        assert_eq!(
            store.checkpoint().active().expect("active").phase,
            AttemptPhase::Generating
        );
    }

    #[test]
    fn uncertain_commit_blocks_until_disk_reconciliation() {
        let dir = TempDir::new().expect("dir");
        let mut store = admitted_store(&dir);
        store.inject_fault(WriteFault::AfterRename);
        assert!(matches!(
            store.begin_generation(),
            Err(CheckpointError::Uncertain(_))
        ));
        assert!(store.is_uncertain());
        assert_eq!(
            store.checkpoint().active().expect("active").phase,
            AttemptPhase::Admitted
        );
        assert!(matches!(
            store.admit_attempt("attempt-2"),
            Err(CheckpointError::UncertainState(_))
        ));
        assert!(matches!(
            store.fail_attempt("must not be recorded"),
            Err(CheckpointError::UncertainState(_))
        ));
        assert!(matches!(
            store.resume(),
            Err(CheckpointError::UncertainState(_))
        ));
        assert!(matches!(
            store.reconcile_from_disk().expect("disk"),
            Recovery::Blocked(_)
        ));
        assert!(!store.is_uncertain());
        assert_eq!(
            store.checkpoint().active().expect("active").phase,
            AttemptPhase::Generating
        );
        assert_eq!(store.checkpoint().consumed, 1);
    }

    #[test]
    fn reconciliation_is_separate_from_history_and_survives_reopen() {
        let dir = TempDir::new().expect("dir");
        {
            let mut store = admitted_store(&dir);
            store.begin_generation().expect("gen");
            store.fail_attempt("original failure").expect("fail");
            store.admit_attempt("attempt-2").expect("admit 2");
            store.begin_generation().expect("gen 2");
            store.block("operator needed").expect("block");
        }
        let mut store = CheckpointStore::open(dir.path(), "run-1").expect("open");
        assert_eq!(
            store.checkpoint().block_reason.as_deref(),
            Some("operator needed")
        );
        assert!(matches!(store.resume().expect("resume"), Recovery::Blocked(_)));
        store.reconcile_abandon(confirmation()).expect("reconcile");
        assert_eq!(store.checkpoint().attempts.len(), 2);
        assert_eq!(
            store.checkpoint().attempts[0].failure.as_deref(),
            Some("original failure")
        );
        assert_eq!(
            store.checkpoint().attempts[1].phase,
            AttemptPhase::Abandoned
        );
        assert!(store.checkpoint().attempts[1].reconciliation.is_some());
        assert_eq!(store.checkpoint().consumed, 2);
        drop(store);

        let mut store = CheckpointStore::open(dir.path(), "run-1").expect("reopen");
        assert!(matches!(store.resume().expect("resume"), Recovery::Ready));
        assert_eq!(
            store.checkpoint().attempts[0].failure.as_deref(),
            Some("original failure")
        );
        assert!(store.checkpoint().attempts[1].reconciliation.is_some());
        assert_eq!(store.checkpoint().consumed, 2);
        store.admit_attempt("attempt-3").expect("admit 3");
        assert_eq!(store.checkpoint().consumed, 3);
    }

    fn provider_binding(model: &str, digest: &str) -> ProviderBinding {
        ProviderBinding {
            provider: "deepseek".into(),
            model: model.into(),
            endpoint: "https://example.test/v1/chat/completions".into(),
            effective_config_sha256: digest.into(),
        }
    }

    #[test]
    fn usage_kind_survives_reopen_without_becoming_actual() {
        let dir = TempDir::new().expect("dir");
        {
            let mut store = admitted_store(&dir);
            store
                .record_usage(Usage::Estimated {
                    input_tokens: 10,
                    output_tokens: 4,
                })
                .expect("usage");
        }
        let store = CheckpointStore::open(dir.path(), "run-1").expect("reopen");
        assert_eq!(
            store.checkpoint().attempts[0].usage,
            Some(Usage::Estimated {
                input_tokens: 10,
                output_tokens: 4
            })
        );
        assert_ne!(
            store.checkpoint().attempts[0].usage,
            Some(Usage::Actual {
                input_tokens: 10,
                output_tokens: 4
            })
        );
    }

    #[test]
    fn binding_cannot_be_added_after_attempt_history() {
        let dir = TempDir::new().expect("dir");
        let mut store = admitted_store(&dir);
        assert!(matches!(
            store.bind_provider(&provider_binding("model-a", "digest-a")),
            Err(CheckpointError::ConfigurationMismatch(_))
        ));
    }

    #[test]
    fn provider_binding_is_immutable_and_copied_to_attempts() {
        let dir = TempDir::new().expect("dir");
        let binding = provider_binding("model-a", "digest-a");
        {
            let mut store = CheckpointStore::create(dir.path(), "run-1", &tree(&[("a", "1")]))
                .expect("create");
            store.bind_provider(&binding).expect("bind before admission");
            store.admit_attempt("attempt-1").expect("admit 1");
            store.fail_attempt("first").expect("fail 1");
            store.admit_attempt("attempt-2").expect("admit 2");
            assert_eq!(
                store.checkpoint().attempts[0].provider.as_ref(),
                Some(&binding)
            );
            assert_eq!(
                store.checkpoint().attempts[1].provider.as_ref(),
                Some(&binding)
            );
            assert_eq!(
                store.checkpoint().provider.as_ref(),
                Some(&binding)
            );
            assert!(store
                .bind_provider(&provider_binding("model-b", "digest-b"))
                .is_err());
        }
        let mut store = CheckpointStore::open(dir.path(), "run-1").expect("reopen");
        assert_eq!(store.checkpoint().provider.as_ref(), Some(&binding));
        assert_eq!(
            store.checkpoint().attempts[0].provider.as_ref(),
            Some(&binding)
        );
        assert_eq!(
            store.checkpoint().attempts[1].provider.as_ref(),
            Some(&binding)
        );
        store
            .bind_provider(&binding)
            .expect("idempotent rebind");
        assert!(store
            .bind_provider(&provider_binding("model-b", "digest-b"))
            .is_err());
    }

    #[test]
    fn complete_cleanup_diagnostics_are_per_attempt_and_survive_reopen() {
        let dir = TempDir::new().expect("dir");
        {
            let mut store = admitted_store(&dir);
            let first = PartialOutput {
                stdout: "first-partial".into(),
                stderr: "first-stderr".into(),
                stdout_truncated: true,
                output_incomplete: true,
                cleanup_incomplete: false,
                ..PartialOutput::default()
            };
            store
                .record_cleanup(CleanupState::Complete, Some(&first))
                .expect("first cleanup");
            store.fail_attempt("first failed").expect("first fail");

            store.admit_attempt("attempt-2").expect("second admit");
            let second = PartialOutput {
                stdout: "second-partial".into(),
                stderr: "second-stderr".into(),
                stderr_truncated: true,
                output_incomplete: true,
                cleanup_incomplete: false,
                ..PartialOutput::default()
            };
            store
                .record_cleanup(CleanupState::Complete, Some(&second))
                .expect("second cleanup");
            // A no-diagnostic update must not erase the second attempt's partial.
            store
                .record_cleanup(CleanupState::Complete, None)
                .expect("no diagnostics update");
        }
        let store = CheckpointStore::open(dir.path(), "run-1").expect("reopen");
        assert_eq!(store.checkpoint().consumed, 2);
        assert_eq!(store.checkpoint().attempts.len(), 2);
        let first = store.checkpoint().attempts[0]
            .partial
            .as_ref()
            .expect("first partial");
        assert_eq!(first.stdout, "first-partial");
        assert_eq!(first.stderr, "first-stderr");
        assert!(first.stdout_truncated);
        assert!(first.output_incomplete);
        let second = store.checkpoint().attempts[1]
            .partial
            .as_ref()
            .expect("second partial");
        assert_eq!(second.stdout, "second-partial");
        assert_eq!(second.stderr, "second-stderr");
        assert!(second.stderr_truncated);
        assert!(second.output_incomplete);
        assert_eq!(store.checkpoint().attempts[0].cleanup, CleanupState::Complete);
        assert_eq!(store.checkpoint().attempts[1].cleanup, CleanupState::Complete);
        assert_eq!(store.checkpoint().cleanup, CleanupState::Complete);
    }

    #[test]
    fn unresolved_cleanup_cannot_be_downgraded_or_bypassed() {
        let dir = TempDir::new().expect("dir");
        let mut store = admitted_store(&dir);
        let partial = PartialOutput {
            stdout: "original partial".into(),
            cleanup_incomplete: true,
            cleanup_error: Some("original cleanup error".into()),
            ..PartialOutput::default()
        };
        store
            .record_cleanup(
                CleanupState::Incomplete {
                    error: "original cleanup error".into(),
                },
                Some(&partial),
            )
            .expect("incomplete cleanup");
        assert_eq!(store.checkpoint().status, RunStatus::Blocked);
        assert!(matches!(store.resume().expect("resume"), Recovery::Blocked(_)));

        assert!(store.record_cleanup(CleanupState::Complete, None).is_err());
        assert!(store.cancel(CleanupState::Complete).is_err());
        assert!(store.finish_failed("must not bypass").is_err());
        assert!(store.finish_exhausted().is_err());
        assert!(store.fail_attempt("must not bypass").is_err());
        assert!(store.admit_attempt("attempt-2").is_err());
        assert!(store.promote().is_err());

        store
            .record_cleanup(
                CleanupState::Incomplete {
                    error: "replacement".into(),
                },
                None,
            )
            .expect("no-op update");
        assert_eq!(
            store.checkpoint().cleanup.error(),
            Some("original cleanup error")
        );
        assert_eq!(
            store.checkpoint().active().expect("active").partial.as_ref().map(|p| p.stdout.as_str()),
            Some("original partial")
        );
        assert_eq!(store.checkpoint().consumed, 1);

        store
            .reconcile_abandon(confirmation())
            .expect("explicit reconciliation");
        assert_eq!(store.checkpoint().status, RunStatus::Running);
        assert_eq!(
            store.checkpoint().attempts[0].cleanup.error(),
            Some("original cleanup error")
        );
        assert!(store.checkpoint().attempts[0].reconciliation.is_some());
        assert_eq!(store.checkpoint().consumed, 1);
    }

    #[test]
    fn reconciliation_requires_trusted_confirmation() {
        let dir = TempDir::new().expect("dir");
        let mut store = admitted_store(&dir);
        store.block("interrupted").expect("block");
        let mut confirmation = confirmation();
        confirmation.execution_uncertainty_resolved = false;
        assert!(store.reconcile_abandon(confirmation).is_err());
        assert_eq!(store.checkpoint().status, RunStatus::Blocked);
    }

    #[test]
    fn blocked_and_terminal_states_reject_mutators() {
        let dir = TempDir::new().expect("dir");
        let mut blocked = admitted_store(&dir);
        blocked.block("interrupted").expect("block");
        assert!(blocked.promote().is_err());
        assert!(blocked.fail_attempt("bypass").is_err());
        assert!(blocked.admit_attempt("attempt-2").is_err());
        assert!(blocked.begin_generation().is_err());

        let dir2 = TempDir::new().expect("dir 2");
        let mut completed = admitted_store(&dir2);
        let candidate = tree(&[("a", "2")]);
        completed.begin_generation().expect("gen");
        completed.begin_probe().expect("probe");
        completed.probe_complete(&candidate).expect("probe done");
        completed.begin_evaluation(EVAL_CONFIG).expect("eval");
        let evidence = evidence(&completed);
        completed.evaluation_complete(evidence).expect("evidence");
        completed.promote().expect("promote");
        assert!(completed.promote().is_err());
        assert!(completed.fail_attempt("bypass").is_err());
        assert!(completed.admit_attempt("attempt-2").is_err());
        assert!(completed.block("bypass").is_err());
        assert!(completed.cancel(CleanupState::Complete).is_err());
        assert!(completed.finish_failed("bypass").is_err());
    }

    #[test]
    fn evaluation_evidence_must_match_candidate_and_evaluator() {
        let dir = TempDir::new().expect("dir");
        let mut store = admitted_store(&dir);
        let candidate = tree(&[("a", "2")]);
        store.begin_generation().expect("gen");
        store.begin_probe().expect("probe");
        store.probe_complete(&candidate).expect("probe done");
        store.begin_evaluation(EVAL_CONFIG).expect("eval");

        let mut wrong_candidate = evidence(&store);
        wrong_candidate.candidate_sha256 = "different".into();
        assert!(store.evaluation_complete(wrong_candidate).is_err());

        let mut wrong_evaluator = evidence(&store);
        wrong_evaluator.evaluator_config_sha256 = "different".into();
        assert!(store.evaluation_complete(wrong_evaluator).is_err());

        let mut empty = evidence(&store);
        empty.case_count = 0;
        empty.cases.clear();
        assert!(store.evaluation_complete(empty).is_err());

        let mut inconsistent = evidence(&store);
        inconsistent.failure_count = 1;
        assert!(store.evaluation_complete(inconsistent).is_err());

        assert_eq!(
            store.checkpoint().active().expect("active").phase,
            AttemptPhase::EvaluationInFlight
        );
        let valid = evidence(&store);
        store.evaluation_complete(valid).expect("valid evidence");
    }

    #[test]
    fn completed_run_resumes_without_execution() {
        let dir = TempDir::new().expect("dir");
        {
            let mut store = admitted_store(&dir);
            let candidate = tree(&[("a", "2")]);
            store.begin_generation().expect("gen");
            store.begin_probe().expect("probe");
            store.probe_complete(&candidate).expect("probe done");
            store.begin_evaluation(EVAL_CONFIG).expect("eval");
            let evidence = evidence(&store);
            store.evaluation_complete(evidence).expect("evidence");
            store.promote().expect("promote");
            assert_eq!(store.checkpoint().status, RunStatus::Completed);
        }
        let mut store = CheckpointStore::open(dir.path(), "run-1").expect("open");
        assert!(matches!(store.resume().expect("resume"), Recovery::Completed));
        assert!(store.admit_attempt("attempt-2").is_err());
        assert_eq!(store.checkpoint().consumed, 1);
    }

    #[test]
    fn unevaluated_candidate_cannot_promote() {
        let dir = TempDir::new().expect("dir");
        let mut store = admitted_store(&dir);
        let candidate = tree(&[("a", "2")]);
        store.begin_generation().expect("gen");
        store.begin_probe().expect("probe");
        store.probe_complete(&candidate).expect("probe");
        assert!(store.promote().is_err());
        assert_eq!(store.checkpoint().status, RunStatus::Running);
    }

    #[test]
    fn cancelled_incomplete_cleanup_blocks_until_reconciliation() {
        let dir = TempDir::new().expect("dir");
        {
            let mut store = admitted_store(&dir);
            store
                .cancel(CleanupState::Incomplete {
                    error: "reap stuck".into(),
                })
                .expect("cancel");
            assert_eq!(
                store.checkpoint().status,
                RunStatus::CancelledCleanupIncomplete
            );
        }
        let mut store = CheckpointStore::open(dir.path(), "run-1").expect("open");
        assert!(matches!(store.resume().expect("resume"), Recovery::Blocked(_)));
        assert_eq!(store.checkpoint().consumed, 1);
        store.reconcile_abandon(confirmation()).expect("reconcile");
        assert_eq!(store.checkpoint().status, RunStatus::Running);
        assert_eq!(store.checkpoint().attempts.len(), 1);
        assert_eq!(
            store.checkpoint().attempts[0].phase,
            AttemptPhase::Abandoned
        );
        store.admit_attempt("attempt-2").expect("new attempt");
        assert_eq!(store.checkpoint().consumed, 2);
    }

    #[test]
    fn repeated_recovery_preserves_accounting_and_history() {
        let dir = TempDir::new().expect("dir");
        {
            let mut store = admitted_store(&dir);
            store.begin_generation().expect("gen");
            store.fail_attempt("first failure").expect("fail");
            store.admit_attempt("attempt-2").expect("second");
        }
        for _ in 0..3 {
            let mut store = CheckpointStore::open(dir.path(), "run-1").expect("open");
            assert!(matches!(store.resume().expect("resume"), Recovery::Blocked(_)));
            assert_eq!(store.checkpoint().consumed, 2);
            assert_eq!(store.checkpoint().attempts.len(), 2);
            assert_eq!(store.checkpoint().attempts[0].attempt_id, "attempt-1");
            assert_eq!(
                store.checkpoint().attempts[0].failure.as_deref(),
                Some("first failure")
            );
        }
    }

    #[test]
    fn checkpoint_does_not_contain_environment_secrets() {
        // Synthetic test-only value; it is not a credential and must never be
        // replaced with a real one. This exercises the serialization boundary.
        const SYNTHETIC_CREDENTIAL: &str = "synthetic-test-credential";
        let dir = TempDir::new().expect("dir");
        std::env::set_var("DEEPSEEK_API_KEY", SYNTHETIC_CREDENTIAL);
        {
            let _store = admitted_store(&dir);
        }
        let text = fs::read_to_string(dir.path().join(CURRENT_FILE)).expect("read");
        assert!(!text.contains(SYNTHETIC_CREDENTIAL));
        assert!(!text.contains("DEEPSEEK_API_KEY"));
    }
}
