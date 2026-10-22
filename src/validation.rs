//! Harness surface validation: the Rust port of `harness2/core/tree.py`.
//!
//! The Python reference validates a *proposal* against the harness it was derived
//! from (`recursion.py::_accept_tree`), in this order: `autotrigger`, `validate`,
//! the benchmark adapter's invariants, then the grader-verdict scrub. This module
//! owns the first two steps and the scrub, and leaves adapter invariants to the
//! caller.
//!
//! # Parity notes
//!
//! * Surface constants are the OpenCode substrate (`substrate/opencode.py`), the
//!   primary substrate in the reference release: four always-on files, a 24 000
//!   character cap, `skills/`, `sub_agents/`, `scripts/`, `plugin/`.
//! * Character counts use `chars()`, matching Python's `len` on `str`.
//! * Frontmatter is read from the first 600 characters, matching Python's
//!   `tree[rel][:600]`; fields must start at column 0, matching `^name:`.
//! * **Not ported:** `py_compile` syntax checking of `scripts/*.py`. There is no
//!   Python parser in this crate; `CensusReport::python_syntax_checked` is
//!   `false` so callers cannot mistake a pass for a compile check.
//! * **Deliberate divergence:** Python's `read_tree` silently drops off-surface
//!   paths into a `rejected` list and lets `_run_stage` turn that into a retry.
//!   Here `validate` reports them directly as [`ValidationError::PathNotAllowed`],
//!   which is the same end-to-end verdict for a proposal that writes outside the
//!   surface.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Surface constants (OpenCode)
// ---------------------------------------------------------------------------

/// Files concatenated into every prompt, in order.
pub const ALWAYS_ON: [&str; 4] = [
    "systemprompt.md",
    "guardrails.md",
    "LongTermMEMORY.md",
    "tool_descriptions/tool_guidance.md",
];

/// Combined character cap over all always-on files.
pub const ALWAYS_ON_MAX: usize = 24_000;

/// Why the cap exists; carried into the diagnostic verbatim.
pub const ALWAYS_ON_CAP_REASON: &str =
    "every token here competes with the agent's own output budget";

/// Directory holding `{name}/SKILL.md`.
pub const SKILL_DIR: &str = "skills";

/// Directory holding `{name}/AGENT.md`.
pub const SUBAGENT_DIR: &str = "sub_agents";

/// Directory holding `{name}.py`.
pub const SCRIPT_DIR: &str = "scripts";

/// Directory holding `{name}.ts`.
pub const PLUGIN_DIR: &str = "plugin";

/// File that receives generated trigger lines.
pub const AUTOTRIGGER_FILE: &str = "guardrails.md";

/// `{desc}` is the skill description with trailing periods stripped.
pub const TRIGGER_TEMPLATE: &str = "- When {desc}, load the `{name}` skill.\n";

/// Markers that must never reach a harness file served as evidence.
pub const GRADER_KEYWORDS: [&str; 3] = ["verdict:", "score_raw:", "graded_by:"];

/// Frontmatter is read from the first 600 characters.
pub const HEADER_SCAN_CHARS: usize = 600;

// ---------------------------------------------------------------------------
// HarnessState
// ---------------------------------------------------------------------------

/// A raw file-tree snapshot: relative path to exact file contents.
///
/// This is the serialization boundary of the harness. It is intentionally a
/// map and not a document model: proposals arrive by mutating a materialised
/// directory and reading it back whole, so a snapshot plus a validator is the
/// only shape that does not lose files (including deletions).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HarnessState {
    pub files: BTreeMap<PathBuf, String>,
}

impl HarnessState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, path: impl Into<PathBuf>, contents: impl Into<String>) {
        self.files.insert(path.into(), contents.into());
    }

    pub fn get(&self, path: &str) -> Option<&str> {
        self.files.get(Path::new(path)).map(String::as_str)
    }

    pub fn from_pairs<I, P, S>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (P, S)>,
        P: Into<PathBuf>,
        S: Into<String>,
    {
        Self {
            files: pairs
                .into_iter()
                .map(|(p, s)| (p.into(), s.into()))
                .collect(),
        }
    }
}

impl From<BTreeMap<PathBuf, String>> for HarnessState {
    fn from(files: BTreeMap<PathBuf, String>) -> Self {
        Self { files }
    }
}

// ---------------------------------------------------------------------------
// ProposalDisposition
// ---------------------------------------------------------------------------

/// What actually happened to a proposal.
///
/// The three arms are not cosmetic: `KeptBase` means the model evaluated the
/// current harness and chose to emit it unchanged, which is evidence of a
/// decision. `Fallback` means the proposal was rejected or errored and the
/// caller substituted the previous tree. Collapsing them loses the information
/// the selection and reflection stages read.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ProposalDisposition {
    /// The model changed the tree and the result validates.
    SuccessfulEdit,
    /// The model emitted a byte-identical copy of the tree it was given.
    KeptBase,
    /// Validation or execution failed; the caller rolled back.
    Fallback,
}

impl ProposalDisposition {
    /// Classify without validating: byte-identical trees are `KeptBase`,
    /// anything else is a `SuccessfulEdit` candidate.
    pub fn classify(surface: &HarnessState, proposed: &HarnessState) -> Self {
        Self::classify_maps(&surface.files, &proposed.files)
    }

    /// [`Self::classify`] over raw snapshots, so callers holding maps need not
    /// clone them into [`HarnessState`].
    pub fn classify_maps(
        surface: &BTreeMap<PathBuf, String>,
        proposed: &BTreeMap<PathBuf, String>,
    ) -> Self {
        if surface == proposed {
            Self::KeptBase
        } else {
            Self::SuccessfulEdit
        }
    }

    /// Classify *and* validate, so a rejected edit is reported as `Fallback`.
    ///
    /// A pre-existing defect in `surface` also yields `Fallback`; this mirrors
    /// `_accept_tree`, which validates the proposed tree and not the baseline.
    pub fn evaluate(
        surface: &HarnessState,
        proposed: &HarnessState,
        allow_subagents: bool,
        reserved_scripts: &HashSet<PathBuf>,
    ) -> (Self, Result<CensusReport, ValidationError>) {
        match validate(
            &surface.files,
            &proposed.files,
            allow_subagents,
            reserved_scripts,
        ) {
            Ok(census) => (Self::classify(surface, proposed), Ok(census)),
            Err(err) => (Self::Fallback, Err(err)),
        }
    }

    /// Every disposition costs one rollout.
    ///
    /// This is the `R = 4k` identity: `2k` candidate probes run whether the
    /// stage was edited, kept, or rolled back. `Fallback` is not free, and a
    /// caller that skips its rollout breaks `Ledger::assert_identity`.
    pub fn charges_rollout(self) -> bool {
        true
    }

    /// Whether the tree content came from the model (`Fallback` did not).
    pub fn is_model_authored(self) -> bool {
        matches!(self, Self::SuccessfulEdit | Self::KeptBase)
    }
}

// ---------------------------------------------------------------------------
// CensusReport
// ---------------------------------------------------------------------------

/// What a validated tree contains; the Rust counterpart of Python's census dict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CensusReport {
    pub always_on_chars: usize,
    pub files: usize,
    pub skills: usize,
    pub subagents: usize,
    pub scripts: usize,
    pub plugins: usize,
    pub hooks: bool,
    /// Always `false`: Python syntax checking of scripts is not ported.
    pub python_syntax_checked: bool,
}

// ---------------------------------------------------------------------------
// ValidationError
// ---------------------------------------------------------------------------

/// One grader-output occurrence inside a proposed file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraderHit {
    pub path: PathBuf,
    pub keyword: &'static str,
    /// Byte offset of the first occurrence.
    pub offset: usize,
}

/// A validation failure. Multiple independent problems are grouped in
/// [`ValidationError::Multiple`] so the caller can feed them all back as retry
/// feedback, matching Python's `" | ".join(errs)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidationError {
    /// A path listed in `reserved_scripts` was added, changed, or removed.
    ReservedScriptViolation { paths: Vec<PathBuf> },
    /// Subagents exist or are activated while the deployment forbids them.
    SubagentsDisallowed { reasons: Vec<String> },
    /// A modified string contains a grader verdict marker.
    GraderLeak { hits: Vec<GraderHit> },
    /// A path is outside the editable surface.
    PathNotAllowed { paths: Vec<PathBuf> },
    /// Skill frontmatter is missing or inconsistent.
    SkillFrontmatter { problems: Vec<String> },
    /// A skill is never named in the always-on text.
    UntriggeredSkill { names: Vec<String> },
    /// Subagent frontmatter is missing or inconsistent.
    SubagentFrontmatter { problems: Vec<String> },
    /// A `python scripts/x.py` reference has no matching file.
    DanglingScriptRef { problems: Vec<String> },
    /// The always-on text exceeds the substrate cap.
    AlwaysOnOverCap { chars: usize, cap: usize },
    /// More than one independent problem was found.
    Multiple(Vec<ValidationError>),
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReservedScriptViolation { paths } => write!(
                f,
                "reserved script(s) changed relative to the baseline: {} (reserved scripts are \
                 runtime-owned and immutable)",
                join_paths(paths)
            ),
            Self::SubagentsDisallowed { reasons } => write!(
                f,
                "sub_agents are not available on this deployment: {}",
                reasons.join(", ")
            ),
            Self::GraderLeak { hits } => {
                let rendered: Vec<String> = hits
                    .iter()
                    .map(|h| {
                        format!(
                            "{} contains grader-verdict marker {:?} at byte {}",
                            h.path.display(),
                            h.keyword,
                            h.offset
                        )
                    })
                    .collect();
                write!(f, "{}", rendered.join(" | "))
            }
            Self::PathNotAllowed { paths } => {
                let rendered: Vec<String> = paths
                    .iter()
                    .map(|p| format!("{} is off the editable harness surface", p.display()))
                    .collect();
                write!(f, "{}", rendered.join(" | "))
            }
            Self::SkillFrontmatter { problems } => write!(f, "{}", problems.join(" | ")),
            Self::UntriggeredSkill { names } => {
                let rendered: Vec<String> = names
                    .iter()
                    .map(|name| {
                        format!(
                            "skill '{name}' is never named in the always-on text, so the agent will \
                             not know to invoke it — an untriggered skill is dead weight. Fix it by \
                             adding a trigger line naming `{name}` to systemprompt.md or \
                             guardrails.md, e.g. \"When <the situation this skill handles>, load the \
                             `{name}` skill.\""
                        )
                    })
                    .collect();
                write!(f, "{}", rendered.join(" | "))
            }
            Self::SubagentFrontmatter { problems } => write!(f, "{}", problems.join(" | ")),
            Self::DanglingScriptRef { problems } => write!(f, "{}", problems.join(" | ")),
            Self::AlwaysOnOverCap { chars, cap } => write!(
                f,
                "always-on content is {chars} chars, over the {cap} cap; {ALWAYS_ON_CAP_REASON}"
            ),
            Self::Multiple(errors) => {
                let rendered: Vec<String> = errors.iter().map(ToString::to_string).collect();
                write!(f, "{}", rendered.join(" | "))
            }
        }
    }
}

impl Error for ValidationError {}

fn join_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// Path rules
// ---------------------------------------------------------------------------

/// Normalise a relative path, rejecting escapes. `None` means "off surface".
pub fn normalize_rel(path: &Path) -> Option<String> {
    let raw = path.to_string_lossy().replace('\\', "/");
    if raw.starts_with('/') {
        return None;
    }
    let mut out = String::with_capacity(raw.len());
    for segment in raw.split('/') {
        match segment {
            "" | "." => continue,
            ".." => return None,
            other => {
                if !out.is_empty() {
                    out.push('/');
                }
                out.push_str(other);
            }
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// `[a-z0-9][a-z0-9_-]*`, the component name rule.
fn component_name_ok(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() || c.is_ascii_digit() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
}

/// `[a-z0-9_][a-z0-9_]*`, the script stem rule.
fn script_stem_ok(stem: &str) -> bool {
    let mut chars = stem.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// File names carried inside a skill directory: `[A-Za-z0-9_.-]+`.
fn skill_asset_ok(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

/// Whether a normalised path is on the editable surface.
pub fn is_allowed(rel: &str) -> bool {
    if ALWAYS_ON.contains(&rel) {
        return true;
    }
    let parts: Vec<&str> = rel.split('/').collect();
    match parts.as_slice() {
        [skill, name, "SKILL.md"] if *skill == SKILL_DIR => component_name_ok(name),
        [skill, name, "scripts", file] if *skill == SKILL_DIR => {
            component_name_ok(name) && skill_asset_ok(file)
        }
        [sub, name, "AGENT.md"] if *sub == SUBAGENT_DIR => component_name_ok(name),
        [script, file] if *script == SCRIPT_DIR => {
            file.strip_suffix(".py").is_some_and(script_stem_ok)
        }
        [plugin, file] if *plugin == PLUGIN_DIR => file
            .strip_suffix(".ts")
            .is_some_and(component_name_ok),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Text helpers
// ---------------------------------------------------------------------------

fn header(text: &str) -> &str {
    match text.char_indices().nth(HEADER_SCAN_CHARS) {
        Some((idx, _)) => &text[..idx],
        None => text,
    }
}

/// Read a column-0 `key: value` line from a frontmatter header.
fn field<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    for line in header.lines() {
        let line = line.trim_end();
        if let Some(rest) = line.strip_prefix(key) {
            if let Some(value) = rest.strip_prefix(':') {
                return Some(value.trim());
            }
        }
    }
    None
}

fn truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "true" | "yes" | "on" | "1" | "enabled" | "allow"
    )
}

fn normalized(map: &BTreeMap<PathBuf, String>) -> BTreeMap<String, &String> {
    let mut out = BTreeMap::new();
    for (path, contents) in map {
        if let Some(rel) = normalize_rel(path) {
            out.insert(rel, contents);
        }
    }
    out
}

fn always_on_text(files: &BTreeMap<String, &String>) -> String {
    ALWAYS_ON
        .iter()
        .map(|rel| files.get(*rel).map_or("", |c| c.as_str()))
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn skill_names(files: &BTreeMap<String, &String>) -> Vec<String> {
    let mut out = BTreeSet::new();
    for rel in files.keys() {
        let parts: Vec<&str> = rel.split('/').collect();
        if let [skill, name, "SKILL.md"] = parts.as_slice() {
            if *skill == SKILL_DIR && component_name_ok(name) {
                out.insert((*name).to_string());
            }
        }
    }
    out.into_iter().collect()
}

fn subagent_names(files: &BTreeMap<String, &String>) -> Vec<String> {
    let mut out = BTreeSet::new();
    for rel in files.keys() {
        let parts: Vec<&str> = rel.split('/').collect();
        if let [sub, name, "AGENT.md"] = parts.as_slice() {
            if *sub == SUBAGENT_DIR && component_name_ok(name) {
                out.insert((*name).to_string());
            }
        }
    }
    out.into_iter().collect()
}

/// Extract `python [3] <path>.py` references from a text body.
fn script_refs(text: &str) -> Vec<String> {
    let mut refs = Vec::new();
    let mut expect_ref = false;
    for raw in text.split_whitespace() {
        let token = raw.trim_start_matches(|c: char| !c.is_ascii_alphanumeric() && c != '/');
        if expect_ref {
            let candidate = token.trim_matches(|c: char| {
                matches!(
                    c,
                    '`' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>' | ',' | ';' | ':'
                ) || c == '.'
            });
            if candidate.ends_with(".py")
                && (candidate.starts_with("scripts/")
                    || (candidate.starts_with("skills/") && candidate.contains("/scripts/")))
            {
                refs.push(candidate.to_string());
            }
            expect_ref = false;
            continue;
        }
        if token == "python" || token == "python3" {
            expect_ref = true;
        }
    }
    refs
}

// ---------------------------------------------------------------------------
// autotrigger
// ---------------------------------------------------------------------------

/// One skill that `autotrigger` would (or did) give a trigger line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TriggerRepair {
    pub name: String,
    /// The exact line appended to [`AUTOTRIGGER_FILE`].
    pub trigger_line: String,
    /// True when the skill is absent from `surface` or its `SKILL.md` changed.
    pub newly_declared: bool,
}

fn trigger_line_for(files: &BTreeMap<String, &String>, name: &str) -> String {
    let path = format!("{SKILL_DIR}/{name}/SKILL.md");
    let description = files
        .get(&path)
        .and_then(|body| field(header(body), "description"))
        .map(|d| d.trim_end_matches('.').trim())
        .filter(|d| !d.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(|| format!("the {name} workflow applies"));
    TRIGGER_TEMPLATE
        .replace("{desc}", &description)
        .replace("{name}", name)
}

fn append_trigger(files: &mut BTreeMap<PathBuf, String>, line: &str) {
    let key = PathBuf::from(AUTOTRIGGER_FILE);
    let current = files.get(&key).cloned().unwrap_or_default();
    let updated = if current.trim().is_empty() {
        line.to_string()
    } else {
        format!("{}\n{}", current.trim_end(), line)
    };
    files.insert(key, updated);
}

/// Skills that receive a generated trigger line, in the order Python appends them.
///
/// A skill already named anywhere in the always-on text is skipped, and a line
/// that would push the body over [`ALWAYS_ON_MAX`] is skipped too — Python's
/// two exit conditions, reproduced.
pub fn autotrigger_report(
    surface: &BTreeMap<PathBuf, String>,
    new: &BTreeMap<PathBuf, String>,
) -> Vec<TriggerRepair> {
    let surface_files = normalized(surface);
    let mut scratch = new.clone();
    let mut repairs = Vec::new();

    for name in skill_names(&normalized(new)) {
        let current = normalized(&scratch);
        if always_on_text(&current).contains(&name) {
            continue;
        }
        let line = trigger_line_for(&current, &name);
        let body_chars = always_on_text(&current).chars().count();
        if body_chars + line.chars().count() + 2 > ALWAYS_ON_MAX {
            continue;
        }
        let path = format!("{SKILL_DIR}/{name}/SKILL.md");
        let newly_declared = surface_files.get(&path) != current.get(&path);
        append_trigger(&mut scratch, &line);
        repairs.push(TriggerRepair {
            name,
            trigger_line: line,
            newly_declared,
        });
    }
    repairs
}

/// Names of skills that `autotrigger` would repair; the Python tracking output.
pub fn autotrigger(
    surface: &BTreeMap<PathBuf, String>,
    new: &BTreeMap<PathBuf, String>,
) -> Vec<String> {
    autotrigger_report(surface, new)
        .into_iter()
        .map(|r| r.name)
        .collect()
}

/// Apply the repairs in place, mirroring Python's mutation of the tree.
pub fn apply_autotrigger(
    surface: &BTreeMap<PathBuf, String>,
    new: &mut BTreeMap<PathBuf, String>,
) -> Vec<String> {
    let repairs = autotrigger_report(surface, new);
    for repair in &repairs {
        append_trigger(new, &repair.trigger_line);
    }
    repairs.into_iter().map(|r| r.name).collect()
}

// ---------------------------------------------------------------------------
// validate
// ---------------------------------------------------------------------------

/// Validate a proposed harness tree against the tree it was derived from.
///
/// `surface` is the baseline the proposal started from; `new` is the tree read
/// back from the improvising session. `reserved_scripts` are runtime-owned paths
/// that must not be added, changed, or removed. `allow_subagents` gates both
/// subagent files and subagent activation tags.
pub fn validate(
    surface: &BTreeMap<PathBuf, String>,
    new: &BTreeMap<PathBuf, String>,
    allow_subagents: bool,
    reserved_scripts: &HashSet<PathBuf>,
) -> Result<CensusReport, ValidationError> {
    let base = normalized(surface);
    let files = normalized(new);

    let mut errors: Vec<ValidationError> = Vec::new();

    // 1. Surface policy: every proposed path must be editable.
    let mut off_surface: Vec<PathBuf> = Vec::new();
    for path in new.keys() {
        match normalize_rel(path) {
            Some(rel) if is_allowed(&rel) => {}
            _ => off_surface.push(path.clone()),
        }
    }
    if !off_surface.is_empty() {
        off_surface.sort();
        off_surface.dedup();
        errors.push(ValidationError::PathNotAllowed {
            paths: off_surface,
        });
    }

    // 2. Reserved scripts are immutable: no add, change, or delete.
    let mut reserved_changed: Vec<PathBuf> = Vec::new();
    for reserved in reserved_scripts {
        let Some(rel) = normalize_rel(reserved) else {
            continue;
        };
        let before = base.get(&rel).map(|c| c.as_str());
        let after = files.get(&rel).map(|c| c.as_str());
        let unchanged = matches!((before, after), (Some(a), Some(b)) if a == b);
        if !unchanged && (before.is_some() || after.is_some()) {
            reserved_changed.push(PathBuf::from(rel));
        }
    }
    if !reserved_changed.is_empty() {
        reserved_changed.sort();
        reserved_changed.dedup();
        errors.push(ValidationError::ReservedScriptViolation {
            paths: reserved_changed,
        });
    }

    // 3. Grader verdicts must not appear in any modified string.
    let mut hits: Vec<GraderHit> = Vec::new();
    for (rel, contents) in &files {
        if base.get(rel).map(|c| c.as_str()) == Some(contents.as_str()) {
            continue;
        }
        let lowered = contents.to_ascii_lowercase();
        for keyword in GRADER_KEYWORDS {
            if let Some(offset) = lowered.find(keyword) {
                hits.push(GraderHit {
                    path: PathBuf::from(rel),
                    keyword,
                    offset,
                });
            }
        }
    }
    if !hits.is_empty() {
        hits.sort_by(|a, b| a.path.cmp(&b.path).then(a.offset.cmp(&b.offset)));
        errors.push(ValidationError::GraderLeak { hits });
    }

    // 4. Always-on cap.
    let body = always_on_text(&files);
    let body_chars = body.chars().count();
    if body_chars > ALWAYS_ON_MAX {
        errors.push(ValidationError::AlwaysOnOverCap {
            chars: body_chars,
            cap: ALWAYS_ON_MAX,
        });
    }

    // 5. Skill frontmatter.
    let mut skill_problems: Vec<String> = Vec::new();
    let mut untriggered: Vec<String> = Vec::new();
    for name in skill_names(&files) {
        let rel = format!("{SKILL_DIR}/{name}/SKILL.md");
        let text = files.get(&rel).map_or("", |c| c.as_str());
        let head = header(text);
        if field(head, "description").is_none_or(|d| d.trim().is_empty()) {
            skill_problems.push(format!(
                "{rel} needs a `description:` in its frontmatter, or the skill is dropped \
                 SILENTLY at load"
            ));
        }
        match field(head, "name") {
            Some(declared) if declared == name => {}
            _ => skill_problems.push(format!(
                "{rel} needs frontmatter `name: {name}` matching its directory (a mismatch is \
                 dropped SILENTLY at load)"
            )),
        }
        if !body.contains(&name) {
            untriggered.push(name);
        }
    }
    if !skill_problems.is_empty() {
        errors.push(ValidationError::SkillFrontmatter {
            problems: skill_problems,
        });
    }
    if !untriggered.is_empty() {
        errors.push(ValidationError::UntriggeredSkill { names: untriggered });
    }

    // 6. Subagents: deployment gate, then frontmatter.
    let subs = subagent_names(&files);
    if !allow_subagents {
        let mut reasons: Vec<String> = Vec::new();
        if !subs.is_empty() {
            reasons.push(format!("sub_agents are not available on this deployment: {subs:?}"));
        }
        let mut activated: Vec<String> = Vec::new();
        for (rel, contents) in &files {
            let head = header(contents);
            let activation = ["subagent", "subagents", "allow_subagents", "enable_subagents"]
                .iter()
                .find_map(|key| field(head, key).filter(|v| truthy(v)).map(|v| (*key, v)))
                .or_else(|| {
                    field(head, "mode")
                        .filter(|v| v.eq_ignore_ascii_case("subagent"))
                        .map(|v| ("mode", v))
                });
            if let Some((key, value)) = activation {
                activated.push(format!("{rel} activates `{key}: {value}`"));
            }
        }
        if !activated.is_empty() {
            reasons.extend(activated);
        }
        if !reasons.is_empty() {
            errors.push(ValidationError::SubagentsDisallowed { reasons });
        }
    }

    let mut sub_problems: Vec<String> = Vec::new();
    for name in &subs {
        let rel = format!("{SUBAGENT_DIR}/{name}/AGENT.md");
        let text = files.get(&rel).map_or("", |c| c.as_str());
        let head = header(text);
        if field(head, "description").is_none_or(|d| d.trim().is_empty()) {
            sub_problems.push(format!("{rel} needs a `description:`"));
        }
        if !field(head, "mode").is_some_and(|m| m == "subagent") {
            sub_problems.push(format!("{rel} needs frontmatter `mode: subagent`"));
        }
    }
    if !sub_problems.is_empty() {
        errors.push(ValidationError::SubagentFrontmatter {
            problems: sub_problems,
        });
    }

    // 7. Every script a skill or always-on file invokes must be shipped.
    let available: BTreeSet<&str> = files
        .keys()
        .filter(|rel| {
            let parts: Vec<&str> = rel.split('/').collect();
            matches!(parts.as_slice(), [script, _] if *script == SCRIPT_DIR)
                || matches!(parts.as_slice(), [skill, _, "scripts", _] if *skill == SKILL_DIR)
        })
        .map(String::as_str)
        .collect();
    let reserved_basenames: BTreeSet<String> = reserved_scripts
        .iter()
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .collect();
    let mut dangling: Vec<String> = Vec::new();
    for (rel, contents) in &files {
        let parts: Vec<&str> = rel.split('/').collect();
        let is_skill_md =
            matches!(parts.as_slice(), [skill, _, "SKILL.md"] if *skill == SKILL_DIR);
        if !is_skill_md && !ALWAYS_ON.contains(&rel.as_str()) {
            continue;
        }
        for reference in script_refs(contents) {
            let basename = reference.rsplit('/').next().unwrap_or(&reference);
            if reserved_basenames.contains(basename) {
                continue;
            }
            if !available.contains(reference.as_str()) {
                dangling.push(format!(
                    "{rel} invokes `{reference}` but that file is not in the harness"
                ));
            }
        }
    }
    if !dangling.is_empty() {
        errors.push(ValidationError::DanglingScriptRef { problems: dangling });
    }

    if !errors.is_empty() {
        return Err(if errors.len() == 1 {
            errors.pop().expect("one error")
        } else {
            ValidationError::Multiple(errors)
        });
    }

    Ok(CensusReport {
        always_on_chars: body_chars,
        files: new.len(),
        skills: skill_names(&files).len(),
        subagents: subs.len(),
        scripts: files
            .keys()
            .filter(|rel| {
                let parts: Vec<&str> = rel.split('/').collect();
                matches!(parts.as_slice(), [script, _] if *script == SCRIPT_DIR)
            })
            .count(),
        plugins: files
            .keys()
            .filter(|rel| {
                let parts: Vec<&str> = rel.split('/').collect();
                matches!(parts.as_slice(), [plugin, _] if *plugin == PLUGIN_DIR)
            })
            .count(),
        hooks: false,
        python_syntax_checked: false,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `harness2/demo.py::BASE_SCRIPT`, verbatim.
    const BASE_SCRIPT: &str = r#"import csv
import sys
from decimal import Decimal

with open(sys.argv[1]) as source:
    amounts = [Decimal(row['amount']) for row in csv.DictReader(source)]
total = sum((amount for amount in amounts if amount > 0), Decimal(0))
print(f'Total: {total:.2f}')
"#;

    const SKILL_MD: &str = "---\nname: doc_review\ndescription: Review a document for arithmetic slips.\n---\n\nCheck totals.\n";
    const SKILL_LINE: &str = "- When reviewing a document, load the `doc_review` skill.\n";

    fn empty_reserved() -> HashSet<PathBuf> {
        HashSet::new()
    }

    /// The DemoAdapter base harness: four always-on files, one script.
    /// Python census: `{always_on_chars: 62, n_files: 5, scripts: 1, skills: 0, sub_agents: 0}`.
    fn golden_base() -> HarnessState {
        let mut state = HarnessState::new();
        for rel in ALWAYS_ON {
            state.insert(rel, "");
        }
        state.insert(
            "systemprompt.md",
            "Reconcile the ledger using python scripts/summarize.py.\n",
        );
        state.insert("scripts/summarize.py", BASE_SCRIPT);
        state
    }

    fn golden_skill_edit() -> HarnessState {
        let mut state = golden_base();
        state.insert("skills/doc_review/SKILL.md", SKILL_MD);
        state.insert("guardrails.md", SKILL_LINE);
        state
    }

    #[test]
    fn golden_base_matches_python_census() {
        let base = golden_base();
        let census = validate(&base.files, &base.files, true, &empty_reserved())
            .expect("base harness validates");

        assert_eq!(census.always_on_chars, 62);
        assert_eq!(census.files, 5);
        assert_eq!(census.scripts, 1);
        assert_eq!(census.skills, 0);
        assert_eq!(census.subagents, 0);
        assert_eq!(census.plugins, 0);
        assert!(!census.hooks);
        // Explicit limitation, not a silent success.
        assert!(!census.python_syntax_checked);
    }

    #[test]
    fn golden_skill_edit_matches_python_census() {
        let base = golden_base();
        let edited = golden_skill_edit();
        let census = validate(&base.files, &edited.files, true, &empty_reserved())
            .expect("skill edit validates");

        assert_eq!(census.always_on_chars, 120);
        assert_eq!(census.files, 6);
        assert_eq!(census.skills, 1);
        assert_eq!(census.scripts, 1);
    }

    /// The three proposal outcomes, end to end:
    /// KeptBase (byte-identical), SuccessfulEdit (valid change), Fallback (rejected).
    #[test]
    fn three_dispositions_are_distinguished() {
        let surface = golden_base();

        // KeptBase: the model emits the tree it was handed, unchanged.
        let kept = surface.clone();
        let (disposition, result) =
            ProposalDisposition::evaluate(&surface, &kept, true, &empty_reserved());
        assert_eq!(disposition, ProposalDisposition::KeptBase);
        assert!(result.is_ok());

        // SuccessfulEdit: a new skill plus its trigger line.
        let edited = golden_skill_edit();
        let (disposition, result) =
            ProposalDisposition::evaluate(&surface, &edited, true, &empty_reserved());
        assert_eq!(disposition, ProposalDisposition::SuccessfulEdit);
        assert!(result.is_ok());

        // Fallback: the skill invokes a script the harness does not ship.
        let mut broken = edited.clone();
        broken.insert(
            "skills/doc_review/SKILL.md",
            "---\nname: doc_review\ndescription: Review a document for arithmetic slips.\n---\n\
             Run `python3 scripts/missing.py`.\n",
        );
        let (disposition, result) =
            ProposalDisposition::evaluate(&surface, &broken, true, &empty_reserved());
        assert_eq!(disposition, ProposalDisposition::Fallback);
        assert!(matches!(
            result,
            Err(ValidationError::DanglingScriptRef { .. })
        ));

        // Budget semantics: every arm costs a rollout; only two are model-authored.
        for disposition in [
            ProposalDisposition::SuccessfulEdit,
            ProposalDisposition::KeptBase,
            ProposalDisposition::Fallback,
        ] {
            assert!(disposition.charges_rollout());
        }
        assert!(ProposalDisposition::SuccessfulEdit.is_model_authored());
        assert!(ProposalDisposition::KeptBase.is_model_authored());
        assert!(!ProposalDisposition::Fallback.is_model_authored());
    }

    #[test]
    fn reserved_scripts_are_immutable() {
        let surface = golden_base();
        let reserved: HashSet<PathBuf> = [PathBuf::from("scripts/summarize.py")]
            .into_iter()
            .collect();

        // Unchanged is fine.
        assert!(validate(&surface.files, &surface.files, true, &reserved).is_ok());

        // Changed.
        let mut changed = surface.clone();
        changed.insert("scripts/summarize.py", "print('rewritten')\n");
        assert_eq!(
            validate(&surface.files, &changed.files, true, &reserved),
            Err(ValidationError::ReservedScriptViolation {
                paths: vec![PathBuf::from("scripts/summarize.py")]
            })
        );

        // Deleted.
        let mut deleted = surface.clone();
        deleted.files.remove(Path::new("scripts/summarize.py"));
        assert!(matches!(
            validate(&surface.files, &deleted.files, true, &reserved),
            Err(ValidationError::ReservedScriptViolation { .. })
        ));

        // Added.
        let mut added = surface.clone();
        added.insert("scripts/helper.py", "print('hi')\n");
        let reserved_added: HashSet<PathBuf> = [PathBuf::from("scripts/helper.py")].into_iter().collect();
        assert!(matches!(
            validate(&surface.files, &added.files, true, &reserved_added),
            Err(ValidationError::ReservedScriptViolation { .. })
        ));
    }

    #[test]
    fn subagents_are_gated_by_deployment_and_frontmatter() {
        let surface = golden_base();
        let mut with_subagent = surface.clone();
        with_subagent.insert(
            "sub_agents/analyst/AGENT.md",
            "---\nmode: subagent\ndescription: Normalise one table.\n---\n",
        );

        // Allowed deployment: validates and is counted.
        let census = validate(&surface.files, &with_subagent.files, true, &empty_reserved())
            .expect("subagent allowed");
        assert_eq!(census.subagents, 1);

        // Forbidden deployment: the file alone is enough.
        let err = validate(&surface.files, &with_subagent.files, false, &empty_reserved())
            .expect_err("subagent disallowed");
        assert!(matches!(err, ValidationError::SubagentsDisallowed { .. }));
        assert!(err.to_string().contains("analyst"));

        // Forbidden deployment: an activation tag in a config file is enough.
        let mut tagged = surface.clone();
        tagged.insert("guardrails.md", "subagent: true\n");
        let err = validate(&surface.files, &tagged.files, false, &empty_reserved())
            .expect_err("activation tag disallowed");
        assert!(matches!(err, ValidationError::SubagentsDisallowed { .. }));

        // The same tag is inert when subagents are permitted.
        assert!(validate(&surface.files, &tagged.files, true, &empty_reserved()).is_ok());
    }

    #[test]
    fn subagent_frontmatter_requires_mode() {
        let surface = golden_base();
        let mut state = surface.clone();
        state.insert(
            "sub_agents/analyst/AGENT.md",
            "---\ndescription: Normalise one table.\n---\n",
        );
        let err = validate(&surface.files, &state.files, true, &empty_reserved())
            .expect_err("mode missing");
        assert!(matches!(
            err,
            ValidationError::SubagentFrontmatter { .. }
        ));
        assert!(err.to_string().contains("mode: subagent"));
    }

    #[test]
    fn grader_verdicts_in_modified_files_are_rejected() {
        let surface = golden_base();

        let mut leaked = surface.clone();
        leaked.insert(
            "systemprompt.md",
            "Reconcile the ledger using python scripts/summarize.py.\nverdict: pass\n",
        );
        let err = validate(&surface.files, &leaked.files, true, &empty_reserved())
            .expect_err("grader leak");
        match &err {
            ValidationError::GraderLeak { hits } => {
                assert_eq!(hits.len(), 1);
                assert_eq!(hits[0].keyword, "verdict:");
                assert_eq!(hits[0].path, PathBuf::from("systemprompt.md"));
            }
            other => panic!("expected GraderLeak, got {other:?}"),
        }

        // Case-insensitive, and only modified files are scanned.
        let mut mixed = surface.clone();
        mixed.insert("guardrails.md", "Score_Raw: 0.91\n");
        assert!(matches!(
            validate(&surface.files, &mixed.files, true, &empty_reserved()),
            Err(ValidationError::GraderLeak { .. })
        ));

        let untouched = surface.clone();
        assert!(validate(&untouched.files, &untouched.files, true, &empty_reserved()).is_ok());
    }

    #[test]
    fn skill_frontmatter_rules_match_python() {
        let surface = golden_base();

        let mut no_description = surface.clone();
        no_description.insert("skills/doc_review/SKILL.md", "---\nname: doc_review\n---\n");
        no_description.insert("guardrails.md", "- load `doc_review`\n");
        let err = validate(&surface.files, &no_description.files, true, &empty_reserved())
            .expect_err("description missing");
        assert!(matches!(err, ValidationError::SkillFrontmatter { .. }));
        assert!(err.to_string().contains("description"));

        let mut bad_name = surface.clone();
        bad_name.insert(
            "skills/doc_review/SKILL.md",
            "---\nname: doc-review\ndescription: d\n---\n",
        );
        bad_name.insert("guardrails.md", "- load `doc_review`\n");
        let err = validate(&surface.files, &bad_name.files, true, &empty_reserved())
            .expect_err("name mismatch");
        assert!(err.to_string().contains("name: doc_review"));
    }

    #[test]
    fn untriggered_skill_is_rejected_until_autotrigger_runs() {
        let surface = golden_base();
        let mut state = surface.clone();
        state.insert("skills/doc_review/SKILL.md", SKILL_MD);

        let err = validate(&surface.files, &state.files, true, &empty_reserved())
            .expect_err("untriggered skill");
        assert!(matches!(err, ValidationError::UntriggeredSkill { .. }));

        // Python tracking output for this fixture is exactly ["doc_review"].
        assert_eq!(autotrigger(&surface.files, &state.files), vec!["doc_review"]);

        let names = apply_autotrigger(&surface.files, &mut state.files);
        assert_eq!(names, vec!["doc_review"]);
        assert_eq!(
            state.get("guardrails.md"),
            Some("- When Review a document for arithmetic slips, load the `doc_review` skill.\n")
        );
        assert!(validate(&surface.files, &state.files, true, &empty_reserved()).is_ok());
    }

    #[test]
    fn autotrigger_skips_named_skills_and_overflow() {
        let surface = golden_base();

        // Already named in always-on text: no repair.
        let mut named = surface.clone();
        named.insert("skills/doc_review/SKILL.md", SKILL_MD);
        named.insert("guardrails.md", SKILL_LINE);
        assert!(autotrigger(&surface.files, &named.files).is_empty());

        // No room under the cap: the line is skipped, leaving the skill untriggered.
        let mut tight = surface.clone();
        tight.insert("skills/doc_review/SKILL.md", SKILL_MD);
        tight.insert("systemprompt.md", "x".repeat(ALWAYS_ON_MAX - 10));
        assert!(autotrigger(&surface.files, &tight.files).is_empty());
        assert!(matches!(
            validate(&surface.files, &tight.files, true, &empty_reserved()),
            Err(ValidationError::UntriggeredSkill { .. })
        ));
    }

    #[test]
    fn off_surface_and_escaping_paths_are_rejected() {
        let surface = golden_base();
        let mut state = surface.clone();
        state.insert("AGENTS.md", "runtime-owned\n");
        state.insert("../escape.md", "nope\n");
        let err = validate(&surface.files, &state.files, true, &empty_reserved())
            .expect_err("off-surface paths");
        match err {
            ValidationError::PathNotAllowed { paths } => {
                assert!(paths.contains(&PathBuf::from("AGENTS.md")));
                assert!(paths.contains(&PathBuf::from("../escape.md")));
            }
            other => panic!("expected PathNotAllowed, got {other:?}"),
        }

        // `./` prefixes normalise onto the surface.
        let mut dotted = surface.clone();
        dotted.insert("./scripts/summarize.py", BASE_SCRIPT);
        dotted.files.remove(Path::new("scripts/summarize.py"));
        assert!(validate(&surface.files, &dotted.files, true, &empty_reserved()).is_ok());
    }

    #[test]
    fn always_on_cap_matches_python_character_count() {
        let surface = golden_base();
        let mut state = surface.clone();
        state.insert("systemprompt.md", "x".repeat(24_001));
        let err = validate(&surface.files, &state.files, true, &empty_reserved())
            .expect_err("over cap");
        match err {
            ValidationError::AlwaysOnOverCap { chars, cap } => {
                // Python: 24007 = 24001 + 2 separators x 3.
                assert_eq!(chars, 24_007);
                assert_eq!(cap, 24_000);
            }
            other => panic!("expected AlwaysOnOverCap, got {other:?}"),
        }
    }

    #[test]
    fn multiple_problems_are_aggregated_for_retry_feedback() {
        let surface = golden_base();
        let mut state = surface.clone();
        state.insert("skills/doc_review/SKILL.md", "---\nname: doc_review\n---\n");
        state.insert("AGENTS.md", "off surface\n");

        match validate(&surface.files, &state.files, true, &empty_reserved()) {
            Err(ValidationError::Multiple(errors)) => {
                assert!(errors
                    .iter()
                    .any(|e| matches!(e, ValidationError::PathNotAllowed { .. })));
                assert!(errors
                    .iter()
                    .any(|e| matches!(e, ValidationError::SkillFrontmatter { .. })));
            }
            other => panic!("expected Multiple, got {other:?}"),
        }
    }

    #[test]
    fn script_reference_scanner_handles_prose_punctuation() {
        assert_eq!(
            script_refs("Reconcile using python scripts/summarize.py.\n"),
            vec!["scripts/summarize.py"]
        );
        assert_eq!(
            script_refs("Run `python3 scripts/missing.py`."),
            vec!["scripts/missing.py"]
        );
        assert_eq!(
            script_refs("`python skills/doc_review/scripts/check.py`"),
            vec!["skills/doc_review/scripts/check.py"]
        );
        assert!(script_refs("python is a language").is_empty());
    }

    #[test]
    fn allowed_paths_match_the_opencode_surface() {
        for allowed in [
            "systemprompt.md",
            "guardrails.md",
            "skills/doc_review/SKILL.md",
            "skills/doc_review/scripts/check.py",
            "sub_agents/analyst/AGENT.md",
            "scripts/summarize.py",
            "plugin/guard.ts",
        ] {
            assert!(is_allowed(allowed), "{allowed} should be allowed");
        }
        for rejected in [
            "AGENTS.md",
            "skills/DocReview/SKILL.md",
            "skills/doc_review/README.md",
            "scripts/Bad-Name.py",
            "scripts/read_doc.py", // allowed by shape; reservedness is a separate check
            "../escape.md",
        ] {
            let expected = rejected == "scripts/read_doc.py";
            assert_eq!(is_allowed(rejected), expected, "{rejected}");
        }
    }
}
