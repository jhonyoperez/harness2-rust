//! Harness² core types, validation, sandboxing, orchestration, and evaluation.
//!
//! The Python reference lives in `harness2/core/tree.py` (surface rules, `validate`,
//! `autotrigger`) and `recursion.py::_accept_tree` (acceptance ordering).
//!
//! * [`validation`] — surface ACL, frontmatter rules, grader-leak scrubbing.
//! * [`sandbox`] — snapshot materialisation and budgeted probe execution.
//! * [`orchestrator`] — charge → validate → probe → classify for one stage.
//! * [`evaluator`] — runtime-owned fixtures and the protection policy.
//! * [`provider`] — typed provider/model configuration, adapters, and transport.
//!
//! Execution goes through [`execution::Isolation`]. Production uses macOS
//! `sandbox-exec` with a deny-default profile and fails closed when it cannot be
//! applied; test-only host executors are compiled under `cfg(test)`. Containment
//! is implemented but unverified in the nested development environment.

pub mod checkpoint;
pub mod evaluator;
pub mod execution;
#[cfg(test)]
mod test_executor;
pub mod orchestrator;
pub mod provider;
pub mod sandbox;
pub mod validation;
