# harness2-rust

> **Status: experimental.** This is a community Rust port of Google Research's
> **Harness²** architecture. It is not production-ready. Real isolation/containment
> and live provider behavior remain unverified. Use at your own risk.

A strongly typed, memory-safe implementation of task-level and harnessing-level
recursive optimization loops under explicit token and budget boundaries.

## 🏗️ Architecture Stack
The framework is engineered as a dependency-light, four-layer deterministic stack:
1. **Validation Layer (`src/validation.rs`)**: Enforces a strict 24,000-character content limit, isolates frontmatter data boundaries, prevents subagent policy drift, and provides active grader-leak substring scanning.
2. **Sandbox Execution Layer (`src/sandbox.rs`, `src/execution.rs`)**: Materializes raw string file-tree snapshots into short-lived `tempfile` workspaces and delegates every execution to one isolation interface; the implemented (but unverified) backend is macOS `sandbox-exec`.
3. **Orchestration Driver (`src/orchestrator.rs`)**: Enforces state transformations, manages the explicit 3-way `ProposalDisposition` telemetry tracking ledger (`SuccessfulEdit`, `KeptBase`, `Fallback`), and handles runtime resource accounting.
4. **Evaluation Layer (`src/evaluator.rs`, `eval/`)**: Runtime-owned fixtures and expected results loaded outside the candidate tree, plus a protection policy that rejects candidate edits to the evaluator, fixtures, expected results, and policy itself.

### Execution isolation

Candidate execution is delegated to `execution::Isolation`; both the probe and the evaluator go through it. The implemented backend is macOS `sandbox-exec` (Seatbelt) with a deny-default profile: no network, writes confined to the per-run workspace, reads limited to system runtime paths and the workspace, a cleared environment, non-root execution, and `RLIMIT_CPU`/`RLIMIT_FSIZE`/`RLIMIT_NOFILE`. If backend detection fails, execution fails closed; there is no host-Python fallback.

One cancellation context is checked before generation, probe, and evaluation, and immediately before completion/promotion. Cancellation preserves the accepted surface, charges the admitted attempt once, and returns `Cancelled` for changed and unchanged proposals. Each operation has one deadline measured from before launch and enforced across child execution and pipe draining; cleanup has a separate bounded grace. stdout/stderr capture is capped at 1 MiB per stream; timeout, cancellation, overflow, and reader errors retain capped partial output with truncation/incomplete flags and are never promoted. Cleanup signals only the `setsid` process group created for the run, reaps the leader, bounds its own wait, and reports `cleanup_incomplete` while a background reaper retains responsibility. The run report retains the latest partial output and cleanup-incomplete count; an incomplete cleanup stops the run instead of becoming an ordinary retryable fault. A future dropped or aborted before completion starts the same cleanup. The fixed launcher call is synchronous and non-preemptible, but its time counts against the deadline. Model output never selects an executor.

**Verification status:** this development environment is itself wrapped in `sandbox-exec`, and macOS refuses to nest sandboxes (`sandbox_apply: Operation not permitted`). Containment is therefore **unverified here**. Unit tests inject a test-only host executor that is compiled only under `#[cfg(test)]`; the normal library contains no host executor and no feature can select one. The containment tests are `#[ignore]`d; run them on an unsandboxed host with `cargo test --ignored`. They are not proof until they pass there.

### Evaluation fixtures

`eval/expected.json` declares cases, `eval/cases/` holds their inputs, and `eval/protected.json` lists the repository paths candidates may never touch. The bundled case distinguishes a correct signed-sum implementation from a positive-only one.

It is a **single-case smoke fixture, not proof of general correctness**: a candidate that hard-codes the expected stdout passes it, and held-out cases are required before any adoption decision.

Evaluation configuration is **fail-closed**. The binary loads `eval/expected.json` and `eval/protected.json` before creating a model client; missing files, malformed JSON, a missing policy, or an empty case list abort the run. `HARNESS2_EVAL_DIR` selects the directory (default `eval`).

Protection prefixes (`eval`, `src`, `tests`, `Cargo.toml`, `Cargo.lock`, `.git`) are mandatory. `protected.json` may add prefixes but cannot remove any of them.

### Acceptance and feedback

A proposal is **accepted** only when the sandbox probe exits zero *and* the runtime-owned evaluator passes. Rejected attempts — validation, protected paths, probe failure, timeout, evaluator error, or evaluation failure — never replace the accepted surface and never count as successful edits.

An empty or unchanged proposal is not completion. The runtime independently evaluates the current surface; only a verified pass completes the run. Outcomes are distinct: `Completed`, `Failed`, `Blocked` (the same failure repeated), `Exhausted` (budget spent), `Cancelled` (caller token observed with complete cleanup), and `CancelledCleanupIncomplete` (terminal, unsafe to retry).

Each rejection returns bounded, structured feedback for the next request: candidate identity, failure category, and truncated diagnostics. Diagnostics are labelled untrusted, delimited, defanged, and never contain hidden expected outputs.

### Provider/model configuration

`src/provider.rs` defines typed provider/model identity, endpoint, credential
reference, context capacity, maximum output tokens, temperature support, and
explicit reasoning support. Context capacity is separate from output allowance.
Unsupported requested settings fail closed instead of being omitted.

Exactly one provider adapter is selected: DeepSeek, GPT/OpenAI, or an
OpenAI-compatible local endpoint. Their request formats differ, and there is no
fallback, escalation, hidden retry, or second model request on failure.

**DeepSeek contract.** The official Chat Completions contract uses
`max_tokens`, `thinking: {"type": "enabled"|"disabled"}`, and
`reasoning_effort: "low"|"high"|"max"`. | Model | Context | Max output | Accepted reasoning | Off |
|---|---:|---:|---|---|
| `deepseek-flash` | 1,000,000 | 393,216 | high | thinking disabled |
| `deepseek-v4-pro` | 1,000,000 | 393,216 | high | thinking disabled |

This client supports only explicit `off` (`thinking: disabled`) and `high`
(`thinking: enabled` plus `reasoning_effort: high`); it never sends a
`budget_tokens` field. Omitted reasoning configuration defaults to high for
DeepSeek, while an explicit `off` is preserved. The contract states that
temperature has no effect in thinking mode, so an explicit temperature in
high/thinking mode is rejected rather than silently ignored. Unsupported
`max` fails clearly.
[Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/) ·
[Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion)

**OpenAI contract.** OpenAI documents that parameter support differs by model.
This client uses an exact supported-model table, not prefix or `-chat`
matching:

| Model | Context | Max output | Reasoning support | Off behavior | Token parameter |
|---|---:|---:|---|---|---|
| `gpt-4o-mini` | 128,000 | 16,384 | non-reasoning | omit | `max_tokens` |
| `gpt-4o` | 128,000 | 16,384 | non-reasoning | omit | `max_tokens` |
| `gpt-4.1` | 1,047,576 | 32,768 | non-reasoning | omit | `max_tokens` |
| `o1` | 200,000 | 100,000 | low, medium, high | rejected | `max_completion_tokens` |
| `o3` | 200,000 | 100,000 | low, medium, high | rejected | `max_completion_tokens` |
| `gpt-5.5` | 1,050,000 | 128,000 | low, medium, high | `reasoning_effort: none` | `max_completion_tokens` |

Temperature is allowed only for non-reasoning table entries and only when the
mode is Off. `max` reasoning is outside this client's accepted set and fails clearly; it
is never mapped to another level. **DeepSeek MAX is not implemented in this
adapter**; only Off and High are supported. Dated snapshots and any name not in
the table (`mystery-chat`, for example) fail closed before dispatch unless the
operator explicitly sets `HARNESS2_MODEL_PROFILE=custom` and supplies the
complete custom profile.

No live provider request was made during development. Provider behavior is
documentation-verified and mock-tested only; live API behavior, credentials,
rate limits, and model availability remain unverified.

[Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/index.md) ·
[Reasoning guide](https://developers.openai.com/api/docs/guides/reasoning)

| Variable | Meaning |
|---|---|
| `HARNESS2_PROVIDER` | `deepseek`, `openai`, or `openai-compatible` |
| `HARNESS2_ENDPOINT` | Explicit endpoint; only DeepSeek falls back to `DEEPSEEK_URL` |
| `HARNESS2_MODEL` | Explicit model id; only DeepSeek falls back to `DEEPSEEK_MODEL` |
| `HARNESS2_CREDENTIAL_ENV` | Environment-variable name holding the credential; provider default (DeepSeek: `DEEPSEEK_API_KEY`, OpenAI: `OPENAI_API_KEY`, compatible: none) |
| `HARNESS2_MAX_OUTPUT_TOKENS` | Requested output allowance, validated against the model capability |
| `HARNESS2_CONTEXT_WINDOW` | Total context capacity override |
| `HARNESS2_MODEL_MAX_OUTPUT_TOKENS` | Model output-capability override |
| `HARNESS2_SUPPORTS_TEMPERATURE` | Temperature capability override |
| `HARNESS2_MODEL_PROFILE` | `custom` enables an explicit unknown-model profile |
| `HARNESS2_REASONING_SUPPORT` | `none`, `effort`, or `toggle` capability override |
| `HARNESS2_REASONING_VALUES` | Custom profile accepted values (`low,medium,high,max`) |
| `HARNESS2_REASONING_OFF` | Custom profile Off behavior (`omit`, `none`, `disabled`, `unsupported`) |
| `HARNESS2_MAX_TOKENS_PARAMETER` | Custom profile token parameter (`max_tokens` or `max_completion_tokens`) |
| `HARNESS2_TEMPERATURE` | Optional temperature; rejected if unsupported |
| `HARNESS2_REASONING` | `off`, `high`, or `effort:low|medium|high|max`; omitted defaults DeepSeek to high; DeepSeek MAX is not implemented |
| `HARNESS2_REQUEST_TIMEOUT_MS` | Per-request timeout |
| `HARNESS2_MAX_REQUEST_BYTES` | Request body bound |

Credential references are resolved only at the HTTP transport boundary. They
are not placed in provider bindings, checkpoints, request logs, or errors;
endpoint userinfo/query/fragment and non-loopback plaintext HTTP are rejected.
Actual, estimated, and unknown usage are distinct; absent provider usage is
reported as unknown, never fabricated as actual. Usage is request-local and is
persisted before a malformed proposal is rejected.

Context checks use a **bytes/4 heuristic estimate**, not a tokenizer guarantee;
the configured context capacity remains a hard cap for that estimate.
Response bodies are read incrementally and bounded even when a server uses
chunked transfer encoding or omits `Content-Length`.

Legacy `DEEPSEEK_URL`, `DEEPSEEK_MODEL`, and `DEEPSEEK_API_KEY` are read only
when the selected provider is DeepSeek. OpenAI uses its own endpoint/model
defaults and `OPENAI_API_KEY` reference; OpenAI-compatible providers use
`HARNESS2_*` settings and never inherit a DeepSeek credential implicitly.

The non-secret provider/model/effective-configuration digest is recorded in
durable run and attempt records. Recovery refuses a changed configuration
before dispatching any request, and completed or in-flight recovery issues
zero provider requests.

### Durable checkpoints and recovery

`src/checkpoint.rs` provides a versioned `CheckpointStore` for a run directory.
Keep `HARNESS2_CHECKPOINT_DIR` outside this worktree; checkpoints contain
source and partial process output. The repository ignores common runtime
checkpoint directories, but an arbitrary operator-supplied path remains the
operator's responsibility. It records run/attempt IDs, phase, the accepted
surface and its SHA-256
identity, evaluation evidence, admitted/consumed accounting, cancellation
state, and cleanup metadata. Environment credentials are not captured
automatically, and evaluation evidence stores hashes rather than expected
fixture output or raw failed-case stderr. Accepted/candidate source and partial
stdout/stderr are persisted verbatim, however, and can contain sensitive
information.

Writes are temp-file + `fsync` + rename + directory `fsync`; the file is an
envelope carrying the payload's SHA-256. A crash leaves the previous or new
commit, never a torn current file. Missing, truncated, tampered, or
version-mismatched state fails closed with no fallback tree.

Admission and its accounting are committed before generation. Every mutator
uses a staged copy and only replaces in-memory state after a successful
persistence commit; definitely-uncommitted failures leave memory unchanged.
Post-rename directory-sync failures are *uncertain* and block all mutations,
including `resume`, until `reconcile_from_disk` adopts the actual file. If an
admitted operation's transition cannot be saved, the last durable in-flight
state is preserved, the run reports `RecoveryBlocked`, known partial diagnostics
are retained, and automatic finalization stops rather than fabricating a
`Failed`/`Rejected` checkpoint.

Unresolved cleanup is monotonic: ordinary `cancel`, `record_cleanup`,
`finish_failed`, `finish_exhausted`, `fail_attempt`, and admission cannot
downgrade it to success or erase its evidence. Only explicit trusted
reconciliation can clear the barrier, and the original cleanup/error evidence
remains attached to the abandoned attempt. Legitimate complete-cleanup updates
still persist the current attempt's partial diagnostics, while a no-diagnostic
update never erases existing useful diagnostics.

Recovery blocks any admitted, in-flight, evaluation-pending, or
cleanup-incomplete attempt and never replays it automatically;
`CancelledCleanupIncomplete` remains blocked. Reconciliation requires a
trusted [`ReconciliationConfirmation`] affirming that execution uncertainty is
resolved and cleanup was verified; a free-form reason alone is insufficient.
The original failure and cleanup evidence remain in history. A completed run
resumes as completed and cannot admit another attempt. A per-run lock admits
one recovery owner; checkpoints carry no process IDs, and recovery never
signals a saved process.

The `run_loop` commits admission before generation, then generation, probe,
evaluation, cancellation, and promotion boundaries. Accepted-surface promotion
is committed before the run reports `Completed`. Recovery restores consumed
accounting and refuses to run in-flight or uncertain state.

### Limitations

* This project is experimental and not production-ready.
* **Live provider behavior is unverified**: provider adapters are documented and
  mock-tested, but no live provider request was made. Model availability,
  credentials, rate limits, and server behavior may differ.
* **DeepSeek MAX is not implemented** in this adapter; the supported subset is
  Off and High only.
* The evaluator is a single-case smoke fixture, not proof of general correctness; a hard-coded output passes it.
* Checkpoint locks are not auto-broken after a hard crash; an operator must
  confirm no live owner before removing a stale lock.
* Checkpoints store accepted/candidate surface contents and partial
  stdout/stderr; credentials must not be placed in harness files or emitted by
  candidates. SHA-256 envelopes detect corruption, not authentication against a
  writer who can rewrite the file.
* **Containment is implemented but unverified**: this environment cannot nest `sandbox-exec`, so supervision/cleanup tests use deterministic mocks and the backend has not executed candidate code. Process-group cleanup cannot contain descendants that deliberately leave the group.
* The CLI does not yet expose a user cancellation source; the executor token and cleanup path are implemented and mock-tested.
* Feedback is one-shot context only; there is no retry planner.

## 🚀 Getting Started

### Local Offline Execution (via Ollama)
```bash
HARNESS2_PROVIDER=openai-compatible HARNESS2_ENDPOINT=http://localhost:11434/v1/chat/completions HARNESS2_MODEL=deepseek-r1:7b cargo run -- "reconcile the ledger and report the total"
```

### Cloud Endpoint Example (Unverified)
```bash
export DEEPSEEK_API_KEY="your-dedicated-api-key"
export HARNESS2_ENDPOINT="https://api.deepseek.com"
cargo run -- "reconcile the ledger and report the total"
```
This is a configuration example, not a production-readiness claim.

### Provenance and license

The source of record is
[google-research/harness2](https://github.com/google-research/harness2) at
commit `1cfe254e8d25b15f37048c3730134686abe60335` (2026-10-13), which is
Apache-2.0 licensed. The [`LICENSE`](LICENSE) in this repository is copied
verbatim from that upstream source. Upstream supplies no `NOTICE` file; none is
invented here. This port retains the upstream license and links to the upstream
project. Upstream source files carry `Copyright 2026 Google LLC` and
Apache-2.0 headers; that attribution is preserved here. Upstream states that
it is not an officially supported Google product; this port does not claim
official Google support.

The migration map kept outside this repository records that the mapping was
derived by reading the shipped upstream Python sources and that REA's available
routes did not provide a Python source provider; those routes were rejected
rather than used for decompilation. No REA code is included in this
publication candidate. This provenance statement is not a legal audit or legal
clearance.

Dependencies are recorded in `Cargo.lock`; no third-party source is vendored in
this repository. See each dependency's own license for its terms.

This project is licensed under the Apache License, Version 2.0. See
[`LICENSE`](LICENSE).
