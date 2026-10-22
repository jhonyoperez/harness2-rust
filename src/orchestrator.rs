//! The Propose → Probe → Compose stage driver.
//!
//! This is the Rust counterpart of `recursion.py::_run_stage` plus
//! `_accept_tree`, reduced to the single-stage primitive the orchestrator loop
//! repeats `3k` times (k INV + k DOM + k compositions):
//!
//! 1. **Charge** the rollout budget. The charge happens *before* anything can
//!    fail, because the `R = 4k` ledger counts every stage — including the ones
//!    that fall back. A rejected proposal still costs its probe in the Python
//!    runner (`_run_stage` rolls the fallback tree), so charging on validation
//!    failure is the invariant, not an oversight.
//! 2. **Validate** the proposal against the current surface. A rejection is a
//!    normal outcome: the stage returns [`ProposalDisposition::Fallback`] with
//!    the diagnostics attached to the pipeline, and no probe runs.
//! 3. **Probe** the accepted tree in an isolated [`SandboxWorkspace`].
//! 4. **Classify** the result: byte-identical trees are `KeptBase`, everything
//!    else is a `SuccessfulEdit`. Probe exit status does not change the
//!    disposition — a valid harness that happens to fail its probe is still the
//!    edit the model made, exactly as `RunView.ok == false` does not un-adopt a
//!    tree in the Python runner.
//!
//! # Why validation errors are not `Err`
//!
//! [`OrchestratorError`] is reserved for conditions the loop cannot continue
//! through: budget exhaustion, a workspace that cannot be materialised, a
//! misconfigured probe (missing script, unwritable pipe), or a timeout. A
//! rejected harness is data the next prompt consumes, so it travels as
//! `Ok((Fallback, None))` and is retained on the pipeline for the caller to
//! read. `step_stage` clears that slot at the start of every stage, so it always
//! describes the stage that just ran.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use crate::execution::{CancellationToken, Executor, PartialOutput};
use crate::sandbox::{ProbeResult, SandboxError, SandboxWorkspace};
use crate::validation::{validate, ProposalDisposition, ValidationError};

// ---------------------------------------------------------------------------
// Budget
// ---------------------------------------------------------------------------

/// The rollout budget for one task.
///
/// The published invariant is `R = 4k`: `k` shared base rollouts, `2k` candidate
/// probes (INV and DOM at width `k`), and `k` final probes. [`Self::for_k`]
/// encodes that arithmetic so the ledger and the recursion cannot drift apart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BudgetLedger {
    pub max_rollouts: usize,
    pub rollouts_consumed: usize,
}

impl BudgetLedger {
    pub const fn new(max_rollouts: usize) -> Self {
        Self {
            max_rollouts,
            rollouts_consumed: 0,
        }
    }

    /// `R = 4k`, for `k` candidates per dimension.
    pub const fn for_k(k: usize) -> Self {
        Self::new(4 * k)
    }

    /// Consume one rollout, or fail without consuming when the budget is spent.
    pub fn charge_rollout(&mut self) -> Result<(), OrchestratorError> {
        if self.rollouts_consumed >= self.max_rollouts {
            return Err(OrchestratorError::BudgetExhausted {
                max_rollouts: self.max_rollouts,
                rollouts_consumed: self.rollouts_consumed,
            });
        }
        self.rollouts_consumed += 1;
        Ok(())
    }

    pub fn remaining(&self) -> usize {
        self.max_rollouts.saturating_sub(self.rollouts_consumed)
    }

    pub fn is_exhausted(&self) -> bool {
        self.rollouts_consumed >= self.max_rollouts
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Conditions that stop the pipeline rather than produce a disposition.
#[derive(Debug)]
pub enum OrchestratorError {
    /// No rollout remains; the charge was refused and nothing was consumed.
    BudgetExhausted {
        max_rollouts: usize,
        rollouts_consumed: usize,
    },
    /// The validated tree could not be materialised on disk.
    Workspace { source: std::io::Error },
    /// The probe could not be started or observed (missing script, spawn, I/O).
    Sandbox(SandboxError),
    /// The probe exceeded its wall-clock budget and was killed.
    ProbeTimeout {
        timeout_ms: u64,
        partial: PartialOutput,
    },
}

impl std::fmt::Display for OrchestratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BudgetExhausted {
                max_rollouts,
                rollouts_consumed,
            } => write!(
                f,
                "rollout budget exhausted: {rollouts_consumed}/{max_rollouts} already consumed"
            ),
            Self::Workspace { source } => {
                write!(f, "could not materialise the sandbox workspace: {source}")
            }
            Self::Sandbox(err) => write!(f, "probe failed: {err}"),
            Self::ProbeTimeout { timeout_ms, .. } => {
                write!(f, "probe exceeded its {timeout_ms} ms budget and was killed")
            }
        }
    }
}

impl std::error::Error for OrchestratorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Workspace { source } => Some(source),
            Self::Sandbox(err) => Some(err),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// One task's recurrent adaptation state: the harness currently under edit and
/// the budget spent reaching it.
///
/// `surface` is the *current* tree the next proposal is diffed against — the
/// base harness at the first stage, then the latest checkpoint in sequential
/// mode. It is deliberately public and mutable for that reason.
pub struct RecurrentPipeline {
    pub surface: BTreeMap<PathBuf, String>,
    pub ledger: BudgetLedger,
    last_validation_error: Option<ValidationError>,
    executor: Option<Arc<dyn Executor>>,
    cancel: Option<CancellationToken>,
}

impl RecurrentPipeline {
    pub fn new(surface: BTreeMap<PathBuf, String>, ledger: BudgetLedger) -> Self {
        Self {
            surface,
            ledger,
            last_validation_error: None,
            executor: None,
            cancel: None,
        }
    }

    /// Inject an executor (tests only; production uses `Isolation::shared`).
    pub fn with_executor(mut self, executor: Arc<dyn Executor>) -> Self {
        self.executor = Some(executor);
        self
    }

    /// Build a `k`-candidate pipeline with the `R = 4k` budget already set.
    pub fn for_k(surface: BTreeMap<PathBuf, String>, k: usize) -> Self {
        Self::new(surface, BudgetLedger::for_k(k))
    }

    pub fn with_cancel(mut self, cancel: Option<CancellationToken>) -> Self {
        self.cancel = cancel;
        self
    }

    pub fn cancel_token(&self) -> Option<CancellationToken> {
        self.cancel.clone()
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
    }

    /// Why the most recent stage fell back, if it did.
    ///
    /// Cleared at the start of each [`Self::step_stage`], so it never describes
    /// an earlier stage.
    pub fn last_validation_error(&self) -> Option<&ValidationError> {
        self.last_validation_error.as_ref()
    }

    pub fn remaining_rollouts(&self) -> usize {
        self.ledger.remaining()
    }

    /// Run one proposal through charge → validate → probe → classify.
    ///
    /// Returns `Ok((Fallback, None))` when validation rejects the tree; the
    /// diagnostics are available via [`Self::last_validation_error`]. Returns
    /// `Err` only when the stage cannot proceed at all. In every outcome but
    /// `BudgetExhausted`, one rollout has been consumed.
    pub async fn step_stage(
        &mut self,
        proposal: &BTreeMap<PathBuf, String>,
        script_to_probe: &PathBuf,
        args: &[&str],
        timeout_ms: u64,
        allow_subagents: bool,
        reserved_scripts: &HashSet<PathBuf>,
    ) -> Result<(ProposalDisposition, Option<ProbeResult>), OrchestratorError> {
        // A. Charge first: a stage that fails validation still costs its rollout.
        self.last_validation_error = None;
        self.ledger.charge_rollout()?;

        // B. Validate; a rejection is a disposition, not an infrastructure error.
        if let Err(error) = validate(
            &self.surface,
            proposal,
            allow_subagents,
            reserved_scripts,
        ) {
            self.last_validation_error = Some(error);
            return Ok((ProposalDisposition::Fallback, None));
        }

        // C. Materialise and probe. Any probe failure below is a fault: the tree
        //    was accepted, so losing the probe means the environment is wrong.
        let workspace = match &self.executor {
            Some(executor) => {
                SandboxWorkspace::create_with(proposal, Arc::clone(executor))
                    .map_err(|source| OrchestratorError::Workspace { source })?
            }
            None => SandboxWorkspace::create(proposal)
                .map_err(|source| OrchestratorError::Workspace { source })?,
        };
        let workspace = workspace.with_cancel(self.cancel.clone());

        let probe = match workspace
            .run_probe(script_to_probe, args, timeout_ms)
            .await
        {
            Ok(result) => result,
            Err(SandboxError::Timeout {
                timeout_ms,
                partial,
                ..
            }) => {
                return Err(OrchestratorError::ProbeTimeout {
                    timeout_ms,
                    partial,
                });
            }
            Err(error) => return Err(OrchestratorError::Sandbox(error)),
        };

        // D. Classify the proposal, not the probe's exit status.
        let disposition = ProposalDisposition::classify_maps(&self.surface, proposal);
        Ok((disposition, Some(probe)))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_executor::HostExecutor;
    use crate::validation::ALWAYS_ON;

    const SUMMARIZE: &str = "print('ok')\n";
    const TRIGGER: &str = "- When reviewing a document, load the `doc_review` skill.\n";
    const SKILL: &str = "---\nname: doc_review\ndescription: Review a document for arithmetic slips.\n---\n";

    fn empty_reserved() -> HashSet<PathBuf> {
        HashSet::new()
    }

    /// The minimal harness that validates: four always-on files, one script
    /// referenced from the system prompt.
    fn base_surface() -> BTreeMap<PathBuf, String> {
        let mut files = BTreeMap::new();
        for rel in ALWAYS_ON {
            files.insert(PathBuf::from(rel), String::new());
        }
        files.insert(
            PathBuf::from("systemprompt.md"),
            "Reconcile the ledger using python scripts/summarize.py.\n".to_string(),
        );
        files.insert(PathBuf::from("scripts/summarize.py"), SUMMARIZE.to_string());
        files
    }

    /// A valid edit: one skill plus the trigger line that names it.
    fn valid_edit() -> BTreeMap<PathBuf, String> {
        let mut files = base_surface();
        files.insert(PathBuf::from("skills/doc_review/SKILL.md"), SKILL.to_string());
        files.insert(PathBuf::from("guardrails.md"), TRIGGER.to_string());
        files
    }

    /// A rejected edit: the skill is never named in the always-on text.
    fn untriggered_edit() -> BTreeMap<PathBuf, String> {
        let mut files = base_surface();
        files.insert(PathBuf::from("skills/doc_review/SKILL.md"), SKILL.to_string());
        files
    }

    fn probe_script() -> PathBuf {
        PathBuf::from("scripts/summarize.py")
    }

    #[test]
    fn ledger_encodes_the_r_equals_4k_invariant() {
        assert_eq!(BudgetLedger::for_k(1).max_rollouts, 4);
        assert_eq!(BudgetLedger::for_k(2).max_rollouts, 8);
        assert_eq!(BudgetLedger::for_k(3).max_rollouts, 12);
    }

    #[test]
    fn ledger_refuses_to_overdraw_without_consuming() {
        let mut ledger = BudgetLedger::new(1);
        assert_eq!(ledger.remaining(), 1);
        ledger.charge_rollout().expect("first charge");
        assert!(ledger.is_exhausted());

        match ledger.charge_rollout() {
            Err(OrchestratorError::BudgetExhausted {
                max_rollouts,
                rollouts_consumed,
            }) => {
                assert_eq!(max_rollouts, 1);
                assert_eq!(rollouts_consumed, 1);
            }
            other => panic!("expected BudgetExhausted, got {other:?}"),
        }
        // A refused charge must not move the counter.
        assert_eq!(ledger.rollouts_consumed, 1);
    }

    #[tokio::test]
    async fn kept_base_when_the_proposal_is_byte_identical() {
        let surface = base_surface();
        let mut pipeline = RecurrentPipeline::for_k(surface.clone(), 1).with_executor(Arc::new(HostExecutor));

        let (disposition, probe) = pipeline
            .step_stage(
                &surface,
                &probe_script(),
                &[],
                5_000,
                true,
                &empty_reserved(),
            )
            .await
            .expect("stage");

        assert_eq!(disposition, ProposalDisposition::KeptBase);
        let probe = probe.expect("KeptBase still probes");
        assert_eq!(probe.exit_code, 0);
        assert_eq!(probe.stdout, "ok\n");
        assert_eq!(pipeline.ledger.rollouts_consumed, 1);
        assert!(pipeline.last_validation_error().is_none());
    }

    #[tokio::test]
    async fn successful_edit_when_the_tree_changes_and_validates() {
        let surface = base_surface();
        let edit = valid_edit();
        let mut pipeline = RecurrentPipeline::for_k(surface, 1).with_executor(Arc::new(HostExecutor));

        let (disposition, probe) = pipeline
            .step_stage(&edit, &probe_script(), &[], 5_000, true, &empty_reserved())
            .await
            .expect("stage");

        assert_eq!(disposition, ProposalDisposition::SuccessfulEdit);
        assert!(probe.is_some());
        assert_eq!(pipeline.ledger.rollouts_consumed, 1);
    }

    #[tokio::test]
    async fn rejected_proposal_falls_back_and_still_consumes_budget() {
        let surface = base_surface();
        let bad = untriggered_edit();
        let mut pipeline = RecurrentPipeline::for_k(surface, 1).with_executor(Arc::new(HostExecutor));

        let (disposition, probe) = pipeline
            .step_stage(&bad, &probe_script(), &[], 5_000, true, &empty_reserved())
            .await
            .expect("a rejection is not an error");

        assert_eq!(disposition, ProposalDisposition::Fallback);
        assert!(probe.is_none(), "a rejected tree must not be probed");
        assert_eq!(pipeline.ledger.rollouts_consumed, 1);
        assert!(matches!(
            pipeline.last_validation_error(),
            Some(ValidationError::UntriggeredSkill { .. })
        ));
    }

    #[tokio::test]
    async fn diagnostics_describe_only_the_latest_stage() {
        let surface = base_surface();
        let mut pipeline = RecurrentPipeline::for_k(surface.clone(), 1).with_executor(Arc::new(HostExecutor));

        pipeline
            .step_stage(
                &untriggered_edit(),
                &probe_script(),
                &[],
                5_000,
                true,
                &empty_reserved(),
            )
            .await
            .expect("fallback");
        assert!(pipeline.last_validation_error().is_some());

        pipeline
            .step_stage(&surface, &probe_script(), &[], 5_000, true, &empty_reserved())
            .await
            .expect("kept base");
        assert!(
            pipeline.last_validation_error().is_none(),
            "the previous stage's error must not leak into this one"
        );
    }

    #[tokio::test]
    async fn sandbox_faults_are_errors_and_still_consume_budget() {
        let surface = base_surface();
        let mut pipeline = RecurrentPipeline::for_k(surface.clone(), 1).with_executor(Arc::new(HostExecutor));

        let error = pipeline
            .step_stage(
                &surface,
                &PathBuf::from("scripts/absent.py"),
                &[],
                5_000,
                true,
                &empty_reserved(),
            )
            .await
            .expect_err("missing probe script is a fault");

        assert!(matches!(
            error,
            OrchestratorError::Sandbox(SandboxError::ScriptNotFound { .. })
        ));
        assert_eq!(pipeline.ledger.rollouts_consumed, 1);
        assert!(pipeline.last_validation_error().is_none());
    }

    #[tokio::test]
    async fn timeouts_surface_as_their_own_variant_and_consume_budget() {
        let mut surface = base_surface();
        surface.insert(
            PathBuf::from("scripts/slow.py"),
            "import time\ntime.sleep(5)\n".to_string(),
        );
        let mut pipeline = RecurrentPipeline::for_k(surface.clone(), 1).with_executor(Arc::new(HostExecutor));

        let error = pipeline
            .step_stage(
                &surface,
                &PathBuf::from("scripts/slow.py"),
                &[],
                100,
                true,
                &empty_reserved(),
            )
            .await
            .expect_err("probe must time out");

        assert!(matches!(
            error,
            OrchestratorError::ProbeTimeout { timeout_ms: 100, .. }
        ));
        assert_eq!(pipeline.ledger.rollouts_consumed, 1);
    }

    #[tokio::test]
    async fn exhausted_budget_stops_the_stage_before_any_work() {
        let surface = base_surface();
        let mut pipeline = RecurrentPipeline::new(surface.clone(), BudgetLedger::new(0)).with_executor(Arc::new(HostExecutor));

        let error = pipeline
            .step_stage(
                &surface,
                &probe_script(),
                &[],
                5_000,
                true,
                &empty_reserved(),
            )
            .await
            .expect_err("no budget");

        assert!(matches!(
            error,
            OrchestratorError::BudgetExhausted {
                max_rollouts: 0,
                rollouts_consumed: 0
            }
        ));
        assert_eq!(pipeline.ledger.rollouts_consumed, 0);
        assert!(pipeline.last_validation_error().is_none());
    }

    /// A full `k = 1` run: base → INV edit → rejected DOM → composition, then a
    /// fifth attempt that must be refused. `R = 4` is spent exactly.
    #[tokio::test]
    async fn multi_turn_run_preserves_transitions_and_exhausts_the_budget() {
        let surface = base_surface();
        let edit = valid_edit();

        let mut composed = edit.clone();
        composed.insert(
            PathBuf::from("systemprompt.md"),
            "Reconcile the ledger carefully using python scripts/summarize.py.\n".to_string(),
        );

        let mut pipeline = RecurrentPipeline::for_k(surface.clone(), 1).with_executor(Arc::new(HostExecutor));
        assert_eq!(pipeline.ledger.max_rollouts, 4);

        let (base_disposition, base_probe) = pipeline
            .step_stage(&surface, &probe_script(), &[], 5_000, true, &empty_reserved())
            .await
            .expect("base stage");
        assert_eq!(base_disposition, ProposalDisposition::KeptBase);
        assert!(base_probe.is_some());

        let (inv_disposition, _) = pipeline
            .step_stage(&edit, &probe_script(), &[], 5_000, true, &empty_reserved())
            .await
            .expect("inv stage");
        assert_eq!(inv_disposition, ProposalDisposition::SuccessfulEdit);

        let (dom_disposition, dom_probe) = pipeline
            .step_stage(
                &untriggered_edit(),
                &probe_script(),
                &[],
                5_000,
                true,
                &empty_reserved(),
            )
            .await
            .expect("dom stage");
        assert_eq!(dom_disposition, ProposalDisposition::Fallback);
        assert!(dom_probe.is_none());
        assert!(pipeline.last_validation_error().is_some());

        let (compose_disposition, compose_probe) = pipeline
            .step_stage(&composed, &probe_script(), &[], 5_000, true, &empty_reserved())
            .await
            .expect("compose stage");
        assert_eq!(compose_disposition, ProposalDisposition::SuccessfulEdit);
        assert!(compose_probe.is_some());
        assert!(pipeline.last_validation_error().is_none());

        // Four stages, four charged rollouts, zero remaining.
        assert_eq!(pipeline.ledger.rollouts_consumed, 4);
        assert!(pipeline.ledger.is_exhausted());

        let error = pipeline
            .step_stage(&composed, &probe_script(), &[], 5_000, true, &empty_reserved())
            .await
            .expect_err("fifth stage has no budget");
        assert!(matches!(
            error,
            OrchestratorError::BudgetExhausted {
                max_rollouts: 4,
                rollouts_consumed: 4
            }
        ));
        assert_eq!(pipeline.ledger.rollouts_consumed, 4);
    }

    /// Rejected proposals are charged too: two invalid stages exhaust a budget
    /// of two without ever spawning a probe.
    #[tokio::test]
    async fn validation_failures_consume_the_budget() {
        let surface = base_surface();
        // A budget of exactly two, so the third stage is the exhausted one.
        let mut pipeline = RecurrentPipeline::new(surface.clone(), BudgetLedger::new(2)).with_executor(Arc::new(HostExecutor));

        for _ in 0..2 {
            let (disposition, probe) = pipeline
                .step_stage(
                    &untriggered_edit(),
                    &probe_script(),
                    &[],
                    5_000,
                    true,
                    &empty_reserved(),
                )
                .await
                .expect("fallback");
            assert_eq!(disposition, ProposalDisposition::Fallback);
            assert!(probe.is_none());
        }

        assert_eq!(pipeline.ledger.rollouts_consumed, 2);
        assert!(matches!(
            pipeline
                .step_stage(&surface, &probe_script(), &[], 5_000, true, &empty_reserved())
                .await,
            Err(OrchestratorError::BudgetExhausted { .. })
        ));
    }
}
