//! Explicit provider/model configuration, capability-aware request adapters,
//! and offline-testable transport dispatch.
//!
//! One `ProviderClient` has exactly one provider, model, endpoint, and
//! credential reference. There is no fallback, escalation, hidden retry, or
//! implicit model selection. Credential values are resolved only inside the
//! transport implementation and are redacted from diagnostics.

use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::execution::CancellationToken;

pub const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
pub const DEFAULT_MAX_REQUEST_BYTES: usize = 1024 * 1024;
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(180);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    DeepSeek,
    OpenAi,
    OpenAiCompatible,
}

impl ProviderKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DeepSeek => "deepseek",
            Self::OpenAi => "openai",
            Self::OpenAiCompatible => "openai-compatible",
        }
    }

    pub fn parse(value: &str) -> Result<Self, ProviderError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "deepseek" => Ok(Self::DeepSeek),
            "openai" | "gpt" => Ok(Self::OpenAi),
            "openai-compatible" | "local" | "compatible" => Ok(Self::OpenAiCompatible),
            other => Err(ProviderError::Config(format!(
                "unsupported provider kind {other:?}"
            ))),
        }
    }

    pub fn default_capabilities(self) -> Capabilities {
        supported_model(self, match self {
            Self::DeepSeek => "deepseek-flash",
            Self::OpenAi => "gpt-4o-mini",
            Self::OpenAiCompatible => "local-model",
        })
        .expect("built-in model table")
        .capabilities()
    }
}

impl ProviderKind {
    /// Exact supported-model capabilities. No prefix or suffix matching:
    /// aliases, dated snapshots, and invented names are unknown unless the
    /// operator explicitly supplies a custom profile.
    pub fn capabilities_for_model(self, model: &str) -> Result<Capabilities, ProviderError> {
        supported_model(self, model)
            .map(|profile| profile.capabilities())
            .ok_or_else(|| {
                ProviderError::Config(format!(
                    "unsupported {} model {model:?}; use a documented model or HARNESS2_MODEL_PROFILE=custom with explicit capabilities",
                    self.as_str()
                ))
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderIdentity {
    pub provider: ProviderKind,
    pub model: String,
}

impl ProviderIdentity {
    pub fn new(provider: ProviderKind, model: impl Into<String>) -> Result<Self, ProviderError> {
        let model = model.into();
        if model.trim().is_empty() {
            return Err(ProviderError::Config("model id is required".into()));
        }
        Ok(Self { provider, model })
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct CredentialRef {
    env_var: String,
}

impl CredentialRef {
    pub fn new(env_var: impl Into<String>) -> Result<Self, ProviderError> {
        let env_var = env_var.into();
        if env_var.trim().is_empty() {
            return Err(ProviderError::Config(
                "credential reference must name an environment variable".into(),
            ));
        }
        Ok(Self { env_var })
    }

    pub fn env_var(&self) -> &str {
        &self.env_var
    }

    /// Resolve only inside the transport boundary. The returned value is never
    /// formatted into logs, errors, checkpoints, or provider bindings.
    fn resolve(&self) -> Result<Secret, TransportError> {
        let value = std::env::var(&self.env_var).map_err(|_| {
            TransportError::Credential(format!(
                "credential reference {} is not set",
                self.env_var
            ))
        })?;
        if value.trim().is_empty() {
            return Err(TransportError::Credential(format!(
                "credential reference {} is empty",
                self.env_var
            )));
        }
        Ok(Secret(value))
    }
}

impl fmt::Debug for CredentialRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CredentialRef").field(&self.env_var).finish()
    }
}

struct Secret(String);

impl Secret {
    fn expose(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Endpoint {
    raw: String,
    public: String,
}

impl Endpoint {
    pub fn new(raw: impl Into<String>) -> Result<Self, ProviderError> {
        let raw = raw.into();
        let url = reqwest::Url::parse(&raw)
            .map_err(|err| ProviderError::Config(format!("invalid endpoint: {err}")))?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(ProviderError::Config(
                "endpoint must use http or https".into(),
            ));
        }
        if !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err(ProviderError::Config(
                "endpoint must not contain userinfo, query parameters, or fragments".into(),
            ));
        }
        if url.scheme() == "http" && !is_loopback_host(url.host_str()) {
            return Err(ProviderError::Config(
                "non-loopback endpoints must use https".into(),
            ));
        }
        let public = url.to_string();
        Ok(Self { raw, public })
    }

    pub fn as_str(&self) -> &str {
        &self.raw
    }

    pub fn public(&self) -> &str {
        &self.public
    }

    fn is_loopback(&self) -> bool {
        reqwest::Url::parse(&self.raw)
            .ok()
            .is_some_and(|url| is_loopback_host(url.host_str()))
    }
}

fn is_loopback_host(host: Option<&str>) -> bool {
    matches!(host, Some("localhost" | "127.0.0.1" | "0.0.0.0" | "::1"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningSupport {
    Unsupported,
    Effort,
    DeepSeekThinking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
    Max,
}

impl ReasoningEffort {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Max => "max",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningRequest {
    Off,
    Effort(ReasoningEffort),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffMode {
    /// The model cannot disable reasoning.
    Unsupported,
    /// Non-reasoning model: omit reasoning parameters.
    Omit,
    /// OpenAI reasoning_effort: "none".
    ReasoningEffortNone,
    /// DeepSeek thinking: {"type":"disabled"}.
    DeepSeekDisabled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReasoningValues(u8);

impl ReasoningValues {
    pub const NONE: Self = Self(0);
    pub const LOW: u8 = 1 << 0;
    pub const MEDIUM: u8 = 1 << 1;
    pub const HIGH: u8 = 1 << 2;
    pub const MAX: u8 = 1 << 3;

    pub const fn from_bits(bits: u8) -> Self {
        Self(bits & 0b1111)
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn contains(self, effort: ReasoningEffort) -> bool {
        self.0 & Self::bit(effort) != 0
    }

    const fn bit(effort: ReasoningEffort) -> u8 {
        match effort {
            ReasoningEffort::Low => Self::LOW,
            ReasoningEffort::Medium => Self::MEDIUM,
            ReasoningEffort::High => Self::HIGH,
            ReasoningEffort::Max => Self::MAX,
        }
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxTokensParameter {
    MaxTokens,
    MaxCompletionTokens,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capabilities {
    /// Total model context capacity, separate from output allowance.
    pub context_window: usize,
    /// Maximum output tokens the model allows.
    pub max_output_tokens: usize,
    pub reasoning: ReasoningSupport,
    pub reasoning_values: ReasoningValues,
    pub off_mode: OffMode,
    pub temperature: bool,
    pub max_tokens_parameter: MaxTokensParameter,
}

/// Small, documented supported-model table. These are exact model IDs, not
/// prefix families; aliases and dated snapshots are unknown unless configured
/// through an explicit custom profile.
#[derive(Debug, Clone, Copy)]
struct SupportedModel {
    provider: ProviderKind,
    model: &'static str,
    context_window: usize,
    max_output_tokens: usize,
    reasoning: ReasoningSupport,
    reasoning_values: ReasoningValues,
    off_mode: OffMode,
    temperature: bool,
    max_tokens_parameter: MaxTokensParameter,
}

impl SupportedModel {
    fn capabilities(self) -> Capabilities {
        Capabilities {
            context_window: self.context_window,
            max_output_tokens: self.max_output_tokens,
            reasoning: self.reasoning,
            reasoning_values: self.reasoning_values,
            off_mode: self.off_mode,
            temperature: self.temperature,
            max_tokens_parameter: self.max_tokens_parameter,
        }
    }
}

const fn reasoning_values(bits: u8) -> ReasoningValues {
    ReasoningValues::from_bits(bits)
}

const SUPPORTED_MODELS: &[SupportedModel] = &[
    SupportedModel {
        provider: ProviderKind::DeepSeek,
        model: "deepseek-flash",
        context_window: 1_000_000,
        max_output_tokens: 393_216,
        reasoning: ReasoningSupport::DeepSeekThinking,
        reasoning_values: reasoning_values(ReasoningValues::HIGH),
        off_mode: OffMode::DeepSeekDisabled,
        temperature: true,
        max_tokens_parameter: MaxTokensParameter::MaxTokens,
    },
    SupportedModel {
        provider: ProviderKind::DeepSeek,
        model: "deepseek-v4-pro",
        context_window: 1_000_000,
        max_output_tokens: 393_216,
        reasoning: ReasoningSupport::DeepSeekThinking,
        reasoning_values: reasoning_values(ReasoningValues::HIGH),
        off_mode: OffMode::DeepSeekDisabled,
        temperature: true,
        max_tokens_parameter: MaxTokensParameter::MaxTokens,
    },
    SupportedModel {
        provider: ProviderKind::OpenAi,
        model: "gpt-4o-mini",
        context_window: 128_000,
        max_output_tokens: 16_384,
        reasoning: ReasoningSupport::Unsupported,
        reasoning_values: ReasoningValues::NONE,
        off_mode: OffMode::Omit,
        temperature: true,
        max_tokens_parameter: MaxTokensParameter::MaxTokens,
    },
    SupportedModel {
        provider: ProviderKind::OpenAi,
        model: "gpt-4o",
        context_window: 128_000,
        max_output_tokens: 16_384,
        reasoning: ReasoningSupport::Unsupported,
        reasoning_values: ReasoningValues::NONE,
        off_mode: OffMode::Omit,
        temperature: true,
        max_tokens_parameter: MaxTokensParameter::MaxTokens,
    },
    SupportedModel {
        provider: ProviderKind::OpenAi,
        model: "gpt-4.1",
        context_window: 1_047_576,
        max_output_tokens: 32_768,
        reasoning: ReasoningSupport::Unsupported,
        reasoning_values: ReasoningValues::NONE,
        off_mode: OffMode::Omit,
        temperature: true,
        max_tokens_parameter: MaxTokensParameter::MaxTokens,
    },
    SupportedModel {
        provider: ProviderKind::OpenAi,
        model: "o1",
        context_window: 200_000,
        max_output_tokens: 100_000,
        reasoning: ReasoningSupport::Effort,
        reasoning_values: reasoning_values(
            ReasoningValues::LOW | ReasoningValues::MEDIUM | ReasoningValues::HIGH,
        ),
        off_mode: OffMode::Unsupported,
        temperature: false,
        max_tokens_parameter: MaxTokensParameter::MaxCompletionTokens,
    },
    SupportedModel {
        provider: ProviderKind::OpenAi,
        model: "o3",
        context_window: 200_000,
        max_output_tokens: 100_000,
        reasoning: ReasoningSupport::Effort,
        reasoning_values: reasoning_values(
            ReasoningValues::LOW | ReasoningValues::MEDIUM | ReasoningValues::HIGH,
        ),
        off_mode: OffMode::Unsupported,
        temperature: false,
        max_tokens_parameter: MaxTokensParameter::MaxCompletionTokens,
    },
    SupportedModel {
        provider: ProviderKind::OpenAi,
        model: "gpt-5.5",
        context_window: 1_050_000,
        max_output_tokens: 128_000,
        reasoning: ReasoningSupport::Effort,
        reasoning_values: reasoning_values(
            ReasoningValues::LOW | ReasoningValues::MEDIUM | ReasoningValues::HIGH,
        ),
        off_mode: OffMode::ReasoningEffortNone,
        temperature: false,
        max_tokens_parameter: MaxTokensParameter::MaxCompletionTokens,
    },
    SupportedModel {
        provider: ProviderKind::OpenAiCompatible,
        model: "local-model",
        context_window: 32_768,
        max_output_tokens: 4_096,
        reasoning: ReasoningSupport::Unsupported,
        reasoning_values: ReasoningValues::NONE,
        off_mode: OffMode::Omit,
        temperature: true,
        max_tokens_parameter: MaxTokensParameter::MaxTokens,
    },
];

fn supported_model(provider: ProviderKind, model: &str) -> Option<SupportedModel> {
    SUPPORTED_MODELS
        .iter()
        .copied()
        .find(|profile| profile.provider == provider && profile.model == model)
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub identity: ProviderIdentity,
    pub endpoint: Endpoint,
    pub credential: Option<CredentialRef>,
    pub capabilities: Capabilities,
    pub max_output_tokens: usize,
    pub temperature: Option<f32>,
    pub reasoning: ReasoningRequest,
    pub timeout: Duration,
    pub max_request_bytes: usize,
}

impl ModelConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        identity: ProviderIdentity,
        endpoint: Endpoint,
        credential: Option<CredentialRef>,
        capabilities: Capabilities,
        max_output_tokens: usize,
        temperature: Option<f32>,
        reasoning: ReasoningRequest,
        timeout: Duration,
        max_request_bytes: usize,
    ) -> Result<Self, ProviderError> {
        let config = Self {
            identity,
            endpoint,
            credential,
            capabilities,
            max_output_tokens,
            temperature,
            reasoning,
            timeout,
            max_request_bytes,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ProviderError> {
        if self.identity.model.trim().is_empty() {
            return Err(ProviderError::Config("model id is required".into()));
        }
        if self.capabilities.context_window == 0
            || self.capabilities.max_output_tokens == 0
        {
            return Err(ProviderError::Config(
                "capabilities must declare positive context and output limits".into(),
            ));
        }
        if self.max_output_tokens == 0 {
            return Err(ProviderError::Config(
                "max_output_tokens must be positive".into(),
            ));
        }
        if self.max_output_tokens > self.capabilities.max_output_tokens {
            return Err(ProviderError::UnsupportedCapability(format!(
                "requested max_output_tokens {} exceeds model maximum {}",
                self.max_output_tokens, self.capabilities.max_output_tokens
            )));
        }
        if self.temperature.is_some() && !self.capabilities.temperature {
            return Err(ProviderError::UnsupportedCapability(
                "temperature is not supported by the selected model family".into(),
            ));
        }
        if let Some(value) = self.temperature {
            if !(0.0..=2.0).contains(&value) {
                return Err(ProviderError::Config(
                    "temperature must be between 0 and 2".into(),
                ));
            }
        }
        let adapter_reasoning = match self.identity.provider {
            ProviderKind::DeepSeek => ReasoningSupport::DeepSeekThinking,
            ProviderKind::OpenAi => ReasoningSupport::Effort,
            ProviderKind::OpenAiCompatible => ReasoningSupport::Unsupported,
        };
        if self.capabilities.reasoning != ReasoningSupport::Unsupported
            && self.capabilities.reasoning != adapter_reasoning
        {
            return Err(ProviderError::UnsupportedCapability(format!(
                "{} adapter does not implement the configured reasoning capability",
                self.identity.provider.as_str()
            )));
        }
        let adapter_tokens = match self.identity.provider {
            ProviderKind::DeepSeek | ProviderKind::OpenAiCompatible => {
                MaxTokensParameter::MaxTokens
            }
            ProviderKind::OpenAi => self.capabilities.max_tokens_parameter,
        };
        if self.identity.provider != ProviderKind::OpenAi
            && self.capabilities.max_tokens_parameter != adapter_tokens
        {
            return Err(ProviderError::UnsupportedCapability(format!(
                "{} adapter does not implement the configured output-token parameter",
                self.identity.provider.as_str()
            )));
        }
        if self.identity.provider == ProviderKind::OpenAi
            && self.capabilities.reasoning == ReasoningSupport::Effort
            && self.capabilities.max_tokens_parameter != MaxTokensParameter::MaxCompletionTokens
        {
            return Err(ProviderError::UnsupportedCapability(
                "OpenAI reasoning families require max_completion_tokens".into(),
            ));
        }
        match self.reasoning {
            ReasoningRequest::Off => match self.capabilities.off_mode {
                OffMode::Unsupported => {
                    return Err(ProviderError::UnsupportedCapability(format!(
                        "{} cannot disable reasoning; choose a supported effort",
                        self.identity.model
                    )))
                }
                OffMode::Omit => {
                    if self.capabilities.reasoning != ReasoningSupport::Unsupported {
                        return Err(ProviderError::UnsupportedCapability(
                            "omitting a reasoning parameter is only valid for non-reasoning models"
                                .into(),
                        ));
                    }
                }
                OffMode::ReasoningEffortNone => {
                    if self.identity.provider != ProviderKind::OpenAi
                        || self.capabilities.reasoning != ReasoningSupport::Effort
                    {
                        return Err(ProviderError::UnsupportedCapability(
                            "reasoning_effort:none requires OpenAI reasoning support".into(),
                        ));
                    }
                }
                OffMode::DeepSeekDisabled => {
                    if self.identity.provider != ProviderKind::DeepSeek
                        || self.capabilities.reasoning != ReasoningSupport::DeepSeekThinking
                    {
                        return Err(ProviderError::UnsupportedCapability(
                            "thinking-disabled Off requires DeepSeek thinking support".into(),
                        ));
                    }
                }
            },
            ReasoningRequest::Effort(effort) => {
                if !self.capabilities.reasoning_values.contains(effort) {
                    return Err(ProviderError::UnsupportedCapability(format!(
                        "reasoning effort {} is not supported by {}; supported values are {:?}",
                        effort.as_str(),
                        self.identity.model,
                        self.capabilities.reasoning_values
                    )));
                }
                if self.identity.provider == ProviderKind::DeepSeek
                    && effort != ReasoningEffort::High
                {
                    return Err(ProviderError::UnsupportedCapability(format!(
                        "DeepSeek supports only high reasoning (requested {})",
                        effort.as_str()
                    )));
                }
                let adapter_reasoning = match self.identity.provider {
                    ProviderKind::DeepSeek => ReasoningSupport::DeepSeekThinking,
                    ProviderKind::OpenAi => ReasoningSupport::Effort,
                    ProviderKind::OpenAiCompatible => ReasoningSupport::Unsupported,
                };
                if self.capabilities.reasoning != adapter_reasoning {
                    return Err(ProviderError::UnsupportedCapability(format!(
                        "{} adapter does not implement the configured reasoning capability",
                        self.identity.provider.as_str()
                    )));
                }
            }
        }
        if matches!(self.reasoning, ReasoningRequest::Effort(_))
            && self.temperature.is_some()
        {
            return Err(ProviderError::UnsupportedCapability(
                "temperature is not supported in reasoning mode".into(),
            ));
        }
        if !self.endpoint.is_loopback() && self.credential.is_none() {
            return Err(ProviderError::Config(
                "a credential reference is required for a non-loopback endpoint".into(),
            ));
        }
        if self.timeout.is_zero() {
            return Err(ProviderError::Config("timeout must be positive".into()));
        }
        if self.max_request_bytes == 0 {
            return Err(ProviderError::Config(
                "max_request_bytes must be positive".into(),
            ));
        }
        Ok(())
    }

    pub fn validate_messages(&self, system: &str, user: &str) -> Result<(), ProviderError> {
        // bytes/4 is a conservative heuristic, not a tokenizer guarantee.
        let estimated_input_tokens =
            heuristic_token_estimate(system) + heuristic_token_estimate(user) + 4;
        let total = estimated_input_tokens.saturating_add(self.max_output_tokens);
        if total > self.capabilities.context_window {
            return Err(ProviderError::ContextExceeded {
                estimated_input_tokens,
                max_output_tokens: self.max_output_tokens,
                context_window: self.capabilities.context_window,
            });
        }
        Ok(())
    }

    /// Non-secret effective identity bound into durable checkpoints.
    pub fn binding(&self) -> ProviderBinding {
        ProviderBinding {
            provider: self.identity.provider.as_str().to_string(),
            model: self.identity.model.clone(),
            endpoint: self.endpoint.public().to_string(),
            effective_config_sha256: self.effective_config_sha256(),
        }
    }

    pub fn effective_config_sha256(&self) -> String {
        let mut canonical = String::new();
        canonical.push_str("harness2-provider-v1\n");
        canonical.push_str(self.identity.provider.as_str());
        canonical.push('\n');
        canonical.push_str(&self.identity.model);
        canonical.push('\n');
        canonical.push_str(self.endpoint.public());
        canonical.push('\n');
        canonical.push_str(&format!(
            "ctx={} cap_out={} req_out={} reason_support={:?} reason_values={:?} off_mode={:?} reason={:?} temp_support={} max_tokens_param={:?} temp={:?} timeout_ms={} req_bytes={}",
            self.capabilities.context_window,
            self.capabilities.max_output_tokens,
            self.max_output_tokens,
            self.capabilities.reasoning,
            self.capabilities.reasoning_values,
            self.capabilities.off_mode,
            self.reasoning,
            self.capabilities.temperature,
            self.capabilities.max_tokens_parameter,
            self.temperature,
            self.timeout.as_millis(),
            self.max_request_bytes
        ));
        canonical.push('\n');
        if let Some(credential) = &self.credential {
            // The reference name is not a secret; the value never enters here.
            canonical.push_str("credential_ref=");
            canonical.push_str(credential.env_var());
            canonical.push('\n');
        }
        crate::checkpoint::sha256_hex(canonical.as_bytes())
    }
}

fn heuristic_token_estimate(text: &str) -> usize {
    (text.len() + 3) / 4
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderBinding {
    pub provider: String,
    pub model: String,
    /// Sanitized endpoint without userinfo/query/fragment.
    pub endpoint: String,
    pub effective_config_sha256: String,
}

impl ProviderBinding {
    pub fn validate(&self) -> Result<(), ProviderError> {
        if self.provider.trim().is_empty()
            || self.model.trim().is_empty()
            || self.endpoint.trim().is_empty()
            || self.effective_config_sha256.trim().is_empty()
        {
            return Err(ProviderError::Config(
                "provider binding is incomplete".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Usage {
    #[default]
    Unknown,
    Actual {
        input_tokens: u64,
        output_tokens: u64,
    },
    Estimated {
        input_tokens: u64,
        output_tokens: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion {
    pub content: String,
    pub usage: Usage,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportRequest {
    pub endpoint: String,
    pub credential: Option<CredentialRef>,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    pub timeout: Duration,
    pub max_response_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportError {
    Credential(String),
    Cancelled,
    Timeout,
    Network(String),
    TooLarge,
}

#[allow(async_fn_in_trait)]
pub trait Transport: Send + Sync {
    async fn execute(
        &self,
        request: TransportRequest,
        cancel: Option<&CancellationToken>,
    ) -> Result<TransportResponse, TransportError>;
}

#[derive(Debug, Default)]
pub struct HttpTransport;

impl Transport for HttpTransport {
    async fn execute(
        &self,
        request: TransportRequest,
        cancel: Option<&CancellationToken>,
    ) -> Result<TransportResponse, TransportError> {
        if cancel.is_some_and(CancellationToken::is_cancelled) {
            return Err(TransportError::Cancelled);
        }
        let secret = match &request.credential {
            Some(credential) => Some(credential.resolve()?),
            None => None,
        };
        let client = reqwest::Client::builder()
            .timeout(request.timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|err| TransportError::Network(redact(err.without_url().to_string(), None)))?;
        let mut builder = client
            .post(&request.endpoint)
            .header("content-type", "application/json")
            .body(request.body);
        for (name, value) in &request.headers {
            builder = builder.header(name, value);
        }
        if let Some(secret) = &secret {
            builder = builder.bearer_auth(secret.expose());
        }
        let mut response = builder.send().await.map_err(|err| {
            map_reqwest_transport_error(err, secret.as_ref().map(Secret::expose))
        })?;
        let status = response.status().as_u16();
        if response
            .content_length()
            .is_some_and(|length| length > request.max_response_bytes as u64)
        {
            return Err(TransportError::TooLarge);
        }
        let mut body = Vec::new();
        loop {
            let chunk = if let Some(token) = cancel {
                tokio::select! {
                    biased;
                    _ = token.cancelled() => return Err(TransportError::Cancelled),
                    chunk = response.chunk() => chunk,
                }
            } else {
                response.chunk().await
            };
            match chunk {
                Ok(Some(chunk)) => {
                    if body.len().saturating_add(chunk.len()) > request.max_response_bytes {
                        return Err(TransportError::TooLarge);
                    }
                    body.extend_from_slice(&chunk);
                }
                Ok(None) => break,
                Err(err) => {
                    return Err(map_reqwest_transport_error(
                        err,
                        secret.as_ref().map(Secret::expose),
                    ))
                }
            }
        }
        Ok(TransportResponse { status, body })
    }
}

fn map_reqwest_transport_error(err: reqwest::Error, secret: Option<&str>) -> TransportError {
    if err.is_timeout() {
        TransportError::Timeout
    } else {
        TransportError::Network(redact(
            err.without_url().to_string(),
            secret,
        ))
    }
}

fn redact(text: String, secret: Option<&str>) -> String {
    let mut text = text;
    if let Some(secret) = secret {
        if !secret.is_empty() {
            text = text.replace(secret, "[REDACTED]");
        }
    }
    if text.len() > 500 {
        let mut end = 500;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        text.push('…');
    }
    text
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderError {
    Config(String),
    UnsupportedCapability(String),
    ContextExceeded {
        estimated_input_tokens: usize,
        max_output_tokens: usize,
        context_window: usize,
    },
    RequestTooLarge {
        bytes: usize,
        max_bytes: usize,
    },
    MissingCredential(String),
    Cancelled,
    Timeout,
    Authentication,
    RateLimited { message: String },
    Status { status: u16, message: String },
    Malformed(String),
    NoContent,
    Transport(String),
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(reason) => write!(f, "provider configuration error: {reason}"),
            Self::UnsupportedCapability(reason) => {
                write!(f, "unsupported provider capability: {reason}")
            }
            Self::ContextExceeded {
                estimated_input_tokens,
                max_output_tokens,
                context_window,
            } => write!(
                f,
                "request exceeds the context window under the bytes/4 heuristic estimate \
                 (not a guaranteed token bound): estimated input {estimated_input_tokens} + \
                 output {max_output_tokens} > {context_window}"
            ),
            Self::RequestTooLarge { bytes, max_bytes } => {
                write!(f, "request body is {bytes} bytes; limit is {max_bytes}")
            }
            Self::MissingCredential(reason) => write!(f, "credential unavailable: {reason}"),
            Self::Cancelled => write!(f, "provider request was cancelled"),
            Self::Timeout => write!(f, "provider request timed out"),
            Self::Authentication => write!(f, "provider authentication failed"),
            Self::RateLimited { message } => write!(f, "provider rate limited: {message}"),
            Self::Status { status, message } => {
                write!(f, "provider returned HTTP {status}: {message}")
            }
            Self::Malformed(reason) => write!(f, "malformed provider response: {reason}"),
            Self::NoContent => write!(f, "provider response contained no content"),
            Self::Transport(reason) => write!(f, "provider transport failure: {reason}"),
        }
    }
}

impl std::error::Error for ProviderError {}

pub struct ProviderClient<T: Transport> {
    config: ModelConfig,
    transport: T,
}

impl<T: Transport> ProviderClient<T> {
    pub fn new(config: ModelConfig, transport: T) -> Result<Self, ProviderError> {
        config.validate()?;
        Ok(Self { config, transport })
    }

    pub fn binding(&self) -> ProviderBinding {
        self.config.binding()
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub async fn complete(
        &self,
        system: &str,
        user: &str,
        cancel: Option<&CancellationToken>,
    ) -> Result<Completion, ProviderError> {
        self.config.validate_messages(system, user)?;
        let body = build_body(&self.config, system, user);
        let bytes = serde_json::to_vec(&body)
            .map_err(|err| ProviderError::Malformed(format!("request serialization: {err}")))?;
        if bytes.len() > self.config.max_request_bytes {
            return Err(ProviderError::RequestTooLarge {
                bytes: bytes.len(),
                max_bytes: self.config.max_request_bytes,
            });
        }
        if cancel.is_some_and(CancellationToken::is_cancelled) {
            return Err(ProviderError::Cancelled);
        }

        let request = TransportRequest {
            endpoint: self.config.endpoint.as_str().to_string(),
            credential: self.config.credential.clone(),
            headers: Vec::new(),
            body: bytes,
            timeout: self.config.timeout,
            max_response_bytes: MAX_RESPONSE_BYTES,
        };
        let response = if let Some(token) = cancel {
            tokio::select! {
                biased;
                _ = token.cancelled() => return Err(ProviderError::Cancelled),
                result = self.transport.execute(request, cancel) => result,
            }
        } else {
            self.transport.execute(request, None).await
        }
        .map_err(map_transport_error)?;

        if response.status == 401 || response.status == 403 {
            return Err(ProviderError::Authentication);
        }
        if response.status == 429 {
            return Err(ProviderError::RateLimited {
                message: "provider returned 429".into(),
            });
        }
        if !(200..300).contains(&response.status) {
            return Err(ProviderError::Status {
                status: response.status,
                message: "request failed".into(),
            });
        }
        let value: Value = serde_json::from_slice(&response.body)
            .map_err(|err| ProviderError::Malformed(format!("response JSON: {err}")))?;
        let content = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .ok_or(ProviderError::NoContent)?
            .to_string();
        let usage = parse_usage(&value);
        Ok(Completion { content, usage })
    }
}

fn map_transport_error(error: TransportError) -> ProviderError {
    match error {
        TransportError::Credential(reason) => ProviderError::MissingCredential(reason),
        TransportError::Cancelled => ProviderError::Cancelled,
        TransportError::Timeout => ProviderError::Timeout,
        TransportError::Network(message) => ProviderError::Transport(message),
        TransportError::TooLarge => ProviderError::Malformed("response too large".into()),
    }
}

fn build_body(config: &ModelConfig, system: &str, user: &str) -> Value {
    let mut body = json!({
        "model": config.identity.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": false,
    });
    let object = body.as_object_mut().expect("object");
    match config.identity.provider {
        ProviderKind::DeepSeek => {
            object.insert("max_tokens".into(), json!(config.max_output_tokens));
            if let Some(temperature) = config.temperature {
                object.insert("temperature".into(), json!(temperature));
            }
            match config.reasoning {
                ReasoningRequest::Off => {
                    object.insert("thinking".into(), json!({"type": "disabled"}));
                }
                ReasoningRequest::Effort(ReasoningEffort::High) => {
                    object.insert("thinking".into(), json!({"type": "enabled"}));
                    object.insert("reasoning_effort".into(), json!("high"));
                }
                ReasoningRequest::Effort(_) => {
                    // Validation rejects this before a request is built.
                }
            }
        }
        ProviderKind::OpenAi => {
            let parameter = match config.capabilities.max_tokens_parameter {
                MaxTokensParameter::MaxTokens => "max_tokens",
                MaxTokensParameter::MaxCompletionTokens => "max_completion_tokens",
            };
            object.insert(parameter.into(), json!(config.max_output_tokens));
            if let Some(temperature) = config.temperature {
                object.insert("temperature".into(), json!(temperature));
            }
            match config.reasoning {
                ReasoningRequest::Off if config.capabilities.off_mode == OffMode::ReasoningEffortNone => {
                    object.insert("reasoning_effort".into(), json!("none"));
                }
                ReasoningRequest::Off => {}
                ReasoningRequest::Effort(effort) => {
                    object.insert("reasoning_effort".into(), json!(effort.as_str()));
                }
            }
        }
        ProviderKind::OpenAiCompatible => {
            object.insert("max_tokens".into(), json!(config.max_output_tokens));
            if let Some(temperature) = config.temperature {
                object.insert("temperature".into(), json!(temperature));
            }
        }
    }
    body
}

fn parse_usage(value: &Value) -> Usage {
    let usage = value.get("usage");
    if let Some(usage) = usage {
        let input = usage.get("prompt_tokens").and_then(Value::as_u64);
        let output = usage.get("completion_tokens").and_then(Value::as_u64);
        if let (Some(input_tokens), Some(output_tokens)) = (input, output) {
            return Usage::Actual {
                input_tokens,
                output_tokens,
            };
        }
    }
    Usage::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    // Clearly synthetic values used only to exercise redaction/rejection.
    // They are deliberately not credential-shaped and are never real secrets.
    const SYNTHETIC_CREDENTIAL: &str = "synthetic-test-credential";
    const SYNTHETIC_SECRET: &str = "synthetic-test-secret";
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Default)]
    struct MockTransport {
        requests: Mutex<Vec<TransportRequest>>,
        responses: Mutex<VecDeque<Result<TransportResponse, TransportError>>>,
        calls: AtomicUsize,
        delay: Mutex<Option<Duration>>,
    }

    impl MockTransport {
        fn with_response(response: Result<TransportResponse, TransportError>) -> Arc<Self> {
            let transport = Arc::new(Self::default());
            transport
                .responses
                .lock()
                .expect("mock")
                .push_back(response);
            transport
        }

        fn requests(&self) -> Vec<TransportRequest> {
            self.requests.lock().expect("mock").clone()
        }
    }

    impl Transport for Arc<MockTransport> {
        async fn execute(
            &self,
            request: TransportRequest,
            cancel: Option<&CancellationToken>,
        ) -> Result<TransportResponse, TransportError> {
            if cancel.is_some_and(CancellationToken::is_cancelled) {
                return Err(TransportError::Cancelled);
            }
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.requests.lock().expect("mock").push(request);
            let delay = *self.delay.lock().expect("mock");
            if let Some(delay) = delay {
                tokio::time::sleep(delay).await;
            }
            self.responses
                .lock()
                .expect("mock")
                .pop_front()
                .unwrap_or(Err(TransportError::Network("no mock response".into())))
        }
    }

    fn credential() -> Option<CredentialRef> {
        Some(CredentialRef::new("HARNESS2_TEST_KEY").expect("credential"))
    }

    fn model_config(kind: ProviderKind) -> ModelConfig {
        let endpoint = match kind {
            ProviderKind::OpenAiCompatible => "http://127.0.0.1:11434/v1/chat/completions",
            _ => "https://example.test/v1/chat/completions",
        };
        let capabilities = if kind == ProviderKind::OpenAi {
            ProviderKind::OpenAi
                .capabilities_for_model("gpt-5.5")
                .expect("gpt-5.5 profile")
        } else {
            kind.default_capabilities()
        };
        ModelConfig::new(
            ProviderIdentity::new(kind, "test-model").expect("identity"),
            Endpoint::new(endpoint).expect("endpoint"),
            if kind == ProviderKind::OpenAiCompatible {
                None
            } else {
                credential()
            },
            capabilities,
            1024,
            if kind == ProviderKind::OpenAi {
                None
            } else {
                Some(0.0)
            },
            ReasoningRequest::Off,
            Duration::from_secs(5),
            32 * 1024,
        )
        .expect("config")
    }

    fn success(content: &str, usage: bool) -> TransportResponse {
        let mut value = json!({
            "choices": [{"message": {"content": content}}]
        });
        if usage {
            value["usage"] = json!({"prompt_tokens": 11, "completion_tokens": 7});
        }
        TransportResponse {
            status: 200,
            body: serde_json::to_vec(&value).expect("json"),
        }
    }

    #[test]
    fn provider_kind_parsing_is_explicit() {
        assert_eq!(ProviderKind::parse("deepseek").expect("deepseek"), ProviderKind::DeepSeek);
        assert_eq!(ProviderKind::parse("gpt").expect("gpt"), ProviderKind::OpenAi);
        assert_eq!(
            ProviderKind::parse("openai-compatible").expect("local"),
            ProviderKind::OpenAiCompatible
        );
        assert!(ProviderKind::parse("automatic").is_err());
    }

    #[test]
    fn rejects_unsupported_temperature_and_reasoning() {
        let mut config = model_config(ProviderKind::OpenAiCompatible);
        config.temperature = Some(0.5);
        config.capabilities.temperature = false;
        assert!(matches!(
            config.validate(),
            Err(ProviderError::UnsupportedCapability(_))
        ));

        let mut config = model_config(ProviderKind::OpenAiCompatible);
        config.reasoning = ReasoningRequest::Effort(ReasoningEffort::Medium);
        assert!(matches!(
            config.validate(),
            Err(ProviderError::UnsupportedCapability(_))
        ));

        let mut config = model_config(ProviderKind::DeepSeek);
        config.reasoning = ReasoningRequest::Effort(ReasoningEffort::Low);
        assert!(matches!(
            config.validate(),
            Err(ProviderError::UnsupportedCapability(_))
        ));
    }

    #[test]
    fn declares_context_and_output_limits_separately() {
        let config = model_config(ProviderKind::DeepSeek);
        assert_eq!(config.capabilities.context_window, 1_000_000);
        assert_eq!(config.capabilities.max_output_tokens, 393_216);
        assert_eq!(config.max_output_tokens, 1024);

        let mut too_large = config;
        too_large.max_output_tokens = too_large.capabilities.max_output_tokens + 1;
        assert!(matches!(
            too_large.validate(),
            Err(ProviderError::UnsupportedCapability(_))
        ));

        let endpoint = Endpoint::new("https://example.test/v1/chat/completions").expect("endpoint");
        let mut tiny = ModelConfig::new(
            ProviderIdentity::new(ProviderKind::DeepSeek, "m").expect("id"),
            endpoint,
            credential(),
            Capabilities {
                context_window: 100,
                max_output_tokens: 10,
                reasoning: ReasoningSupport::DeepSeekThinking,
                reasoning_values: reasoning_values(ReasoningValues::HIGH),
                off_mode: OffMode::DeepSeekDisabled,
                temperature: false,
                max_tokens_parameter: MaxTokensParameter::MaxTokens,
            },
            10,
            None,
            ReasoningRequest::Off,
            Duration::from_secs(1),
            1024,
        )
        .expect("config");
        let error = tiny
            .validate_messages(&"x".repeat(1000), "")
            .expect_err("context");
        assert!(matches!(error, ProviderError::ContextExceeded { .. }));
        assert!(error.to_string().contains("heuristic"));
        assert!(error.to_string().contains("not a guaranteed token bound"));
        tiny.max_output_tokens = 0;
        assert!(matches!(
            tiny.validate(),
            Err(ProviderError::Config(_))
        ));
    }

    #[test]
    fn documented_model_table_limits_and_off_modes() {
        let chat = ProviderKind::OpenAi
            .capabilities_for_model("gpt-4o-mini")
            .expect("chat profile");
        assert_eq!(chat.context_window, 128_000);
        assert_eq!(chat.max_output_tokens, 16_384);
        assert!(chat.temperature);
        assert_eq!(chat.off_mode, OffMode::Omit);
        assert!(chat.reasoning_values.is_empty());

        let long = ProviderKind::OpenAi
            .capabilities_for_model("gpt-4.1")
            .expect("gpt-4.1");
        assert_eq!(long.context_window, 1_047_576);
        assert_eq!(long.max_output_tokens, 32_768);

        let o1 = ProviderKind::OpenAi
            .capabilities_for_model("o1")
            .expect("o1");
        assert_eq!(o1.context_window, 200_000);
        assert_eq!(o1.max_output_tokens, 100_000);
        assert!(!o1.temperature);
        assert_eq!(o1.off_mode, OffMode::Unsupported);
        assert!(o1.reasoning_values.contains(ReasoningEffort::High));
        assert!(!o1.reasoning_values.contains(ReasoningEffort::Max));

        let gpt55 = ProviderKind::OpenAi
            .capabilities_for_model("gpt-5.5")
            .expect("gpt-5.5");
        assert_eq!(gpt55.context_window, 1_050_000);
        assert_eq!(gpt55.max_output_tokens, 128_000);
        assert_eq!(gpt55.off_mode, OffMode::ReasoningEffortNone);
        assert!(gpt55.reasoning_values.contains(ReasoningEffort::High));
        assert!(!gpt55.reasoning_values.contains(ReasoningEffort::Max));

        assert!(ProviderKind::OpenAi
            .capabilities_for_model("mystery-chat")
            .is_err());
        assert!(ProviderKind::OpenAi
            .capabilities_for_model("gpt-4o-mini-2024-07-18")
            .is_err());

        let deepseek = ProviderKind::DeepSeek
            .capabilities_for_model("deepseek-flash")
            .expect("deepseek");
        assert_eq!(deepseek.context_window, 1_000_000);
        assert_eq!(deepseek.max_output_tokens, 393_216);
        assert_eq!(deepseek.off_mode, OffMode::DeepSeekDisabled);
        assert!(deepseek.reasoning_values.contains(ReasoningEffort::High));
        assert!(!deepseek.reasoning_values.contains(ReasoningEffort::Max));

        let mut chat_config = model_config(ProviderKind::OpenAi);
        chat_config.capabilities = chat;
        chat_config.temperature = Some(0.2);
        chat_config.validate().expect("chat config");
        let body = build_body(&chat_config, "s", "u");
        assert_eq!(body["max_tokens"], 1024);
        assert!(body.get("max_completion_tokens").is_none());
        assert!(body.get("reasoning_effort").is_none());

        let mut none_config = model_config(ProviderKind::OpenAi);
        none_config.capabilities = gpt55;
        none_config.reasoning = ReasoningRequest::Off;
        none_config.validate().expect("none off");
        assert_eq!(
            build_body(&none_config, "s", "u")["reasoning_effort"],
            "none"
        );

        let mut reject_off = model_config(ProviderKind::OpenAi);
        reject_off.capabilities = o1;
        reject_off.reasoning = ReasoningRequest::Off;
        assert!(matches!(
            reject_off.validate(),
            Err(ProviderError::UnsupportedCapability(_))
        ));

        let mut unsupported_max = model_config(ProviderKind::OpenAi);
        unsupported_max.capabilities = ProviderKind::OpenAi
            .capabilities_for_model("gpt-5.5")
            .expect("gpt-5.5");
        unsupported_max.reasoning = ReasoningRequest::Effort(ReasoningEffort::Max);
        let error = unsupported_max.validate().expect_err("max unsupported");
        assert!(error.to_string().contains("max"));
    }

    #[test]
    fn deepseek_high_and_off_are_the_only_reasoning_modes() {
        let mut high = model_config(ProviderKind::DeepSeek);
        high.reasoning = ReasoningRequest::Effort(ReasoningEffort::High);
        high.temperature = None;
        high.validate().expect("high");
        let body = build_body(&high, "s", "u");
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["reasoning_effort"], "high");

        let mut off = model_config(ProviderKind::DeepSeek);
        off.reasoning = ReasoningRequest::Off;
        off.validate().expect("off");
        let body = build_body(&off, "s", "u");
        assert_eq!(body["thinking"]["type"], "disabled");
        assert!(body.get("reasoning_effort").is_none());

        let mut low = model_config(ProviderKind::DeepSeek);
        low.reasoning = ReasoningRequest::Effort(ReasoningEffort::Low);
        assert!(matches!(
            low.validate(),
            Err(ProviderError::UnsupportedCapability(_))
        ));

        let mut high_with_temp = high;
        high_with_temp.temperature = Some(0.0);
        assert!(matches!(
            high_with_temp.validate(),
            Err(ProviderError::UnsupportedCapability(_))
        ));
    }

    #[test]
    fn gpt_temperature_is_rejected_for_reasoning_profiles() {
        let mut config = model_config(ProviderKind::OpenAi);
        config.reasoning = ReasoningRequest::Effort(ReasoningEffort::High);
        config.temperature = Some(0.0);
        assert!(matches!(
            config.validate(),
            Err(ProviderError::UnsupportedCapability(_))
        ));
    }

    #[test]
    fn adapters_emit_provider_specific_parameters() {
        let deepseek = model_config(ProviderKind::DeepSeek);
        let body = build_body(&deepseek, "s", "u");
        assert_eq!(body["max_tokens"], 1024);
        assert_eq!(body["thinking"]["type"], "disabled");
        assert!(body.get("reasoning_effort").is_none());
        assert!(body.get("max_completion_tokens").is_none());

        let mut deepseek_high = model_config(ProviderKind::DeepSeek);
        deepseek_high.temperature = None;
        deepseek_high.reasoning = ReasoningRequest::Effort(ReasoningEffort::High);
        deepseek_high.validate().expect("deepseek high");
        let body = build_body(&deepseek_high, "s", "u");
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["reasoning_effort"], "high");
        assert!(body["thinking"].get("budget_tokens").is_none());
        assert!(body.get("temperature").is_none());

        let mut gpt = model_config(ProviderKind::OpenAi);
        gpt.reasoning = ReasoningRequest::Effort(ReasoningEffort::High);
        gpt.validate().expect("gpt config");
        let body = build_body(&gpt, "s", "u");
        assert_eq!(body["max_completion_tokens"], 1024);
        assert_eq!(body["reasoning_effort"], "high");
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("thinking").is_none());

        let mut local = model_config(ProviderKind::OpenAiCompatible);
        local.temperature = Some(0.2);
        local.validate().expect("local config");
        let body = build_body(&local, "s", "u");
        assert_eq!(body["max_tokens"], 1024);
        assert!((body["temperature"].as_f64().expect("temperature") - 0.2).abs() < 1e-6);
        assert!(body.get("reasoning_effort").is_none());
        assert!(body.get("thinking").is_none());
    }

    #[tokio::test]
    async fn pre_cancelled_request_makes_zero_transport_calls() {
        let transport = MockTransport::with_response(Ok(success("{}", true)));
        let client = ProviderClient::new(model_config(ProviderKind::DeepSeek), Arc::clone(&transport))
            .expect("client");
        let token = CancellationToken::new();
        token.cancel();
        let error = client
            .complete("s", "u", Some(&token))
            .await
            .expect_err("cancelled");
        assert_eq!(error, ProviderError::Cancelled);
        assert_eq!(transport.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn cancellation_during_request_returns_cancelled() {
        let transport = MockTransport::with_response(Ok(success("{}", true)));
        *transport.delay.lock().expect("mock") = Some(Duration::from_secs(10));
        let client = ProviderClient::new(model_config(ProviderKind::DeepSeek), Arc::clone(&transport))
            .expect("client");
        let token = CancellationToken::new();
        let clone = token.clone();
        let task = tokio::spawn(async move {
            client.complete("s", "u", Some(&clone)).await
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        token.cancel();
        let result = task.await.expect("task");
        assert_eq!(result, Err(ProviderError::Cancelled));
        assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn timeout_auth_rate_limit_and_malformed_are_distinct() {
        let cases = vec![
            (Err(TransportError::Timeout), ProviderError::Timeout),
            (Ok(TransportResponse { status: 401, body: vec![] }), ProviderError::Authentication),
            (
                Ok(TransportResponse {
                    status: 429,
                    body: b"slow down".to_vec(),
                }),
                ProviderError::RateLimited {
                    message: "provider returned 429".into(),
                },
            ),
            (
                Ok(TransportResponse {
                    status: 200,
                    body: b"not-json".to_vec(),
                }),
                ProviderError::Malformed("response JSON: expected value at line 1 column 1".into()),
            ),
        ];
        for (response, expected) in cases {
            let transport = MockTransport::with_response(response);
            let client =
                ProviderClient::new(model_config(ProviderKind::DeepSeek), Arc::clone(&transport))
                    .expect("client");
            let actual = client.complete("s", "u", None).await.expect_err("error");
            if let ProviderError::Malformed(_) = expected {
                assert!(matches!(actual, ProviderError::Malformed(_)));
            } else {
                assert_eq!(actual, expected);
            }
            assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
        }
    }

    #[tokio::test]
    async fn one_selected_provider_is_used_and_usage_is_distinguished() {
        let transport = MockTransport::with_response(Ok(success("hello", true)));
        let client = ProviderClient::new(model_config(ProviderKind::OpenAi), Arc::clone(&transport))
            .expect("client");
        let completion = client.complete("s", "u", None).await.expect("completion");
        assert_eq!(completion.content, "hello");
        assert_eq!(
            completion.usage,
            Usage::Actual {
                input_tokens: 11,
                output_tokens: 7
            }
        );
        assert_eq!(transport.requests().len(), 1);
        let request: serde_json::Value =
            serde_json::from_slice(&transport.requests()[0].body).expect("request json");
        assert_eq!(request["model"], "test-model");
        assert_eq!(request["max_completion_tokens"], 1024);
    }

    #[tokio::test]
    async fn unknown_usage_is_not_reported_as_actual() {
        let transport = MockTransport::with_response(Ok(success("hello", false)));
        let client = ProviderClient::new(model_config(ProviderKind::DeepSeek), Arc::clone(&transport))
            .expect("client");
        let completion = client.complete("s", "u", None).await.expect("completion");
        assert_eq!(completion.usage, Usage::Unknown);
    }

    #[test]
    fn effective_config_identity_changes_with_request_controls() {
        let base = model_config(ProviderKind::OpenAi);
        let base_digest = base.binding().effective_config_sha256;

        let mut changed = base.clone();
        changed.max_output_tokens -= 1;
        assert_ne!(changed.binding().effective_config_sha256, base_digest);

        let mut changed = base.clone();
        changed.temperature = Some(0.5);
        assert_ne!(changed.binding().effective_config_sha256, base_digest);

        let mut changed = base.clone();
        changed.reasoning = ReasoningRequest::Effort(ReasoningEffort::High);
        assert_ne!(changed.binding().effective_config_sha256, base_digest);

        let mut changed = base.clone();
        changed.identity.model = "different-model".into();
        assert_ne!(changed.binding().effective_config_sha256, base_digest);

        let mut changed = base.clone();
        changed.capabilities.context_window -= 1;
        assert_ne!(changed.binding().effective_config_sha256, base_digest);

        let mut changed = base.clone();
        changed.capabilities.off_mode = OffMode::Omit;
        assert_ne!(changed.binding().effective_config_sha256, base_digest);

        let mut changed = base.clone();
        changed.capabilities.reasoning_values =
            ReasoningValues::from_bits(ReasoningValues::LOW);
        assert_ne!(changed.binding().effective_config_sha256, base_digest);

        let mut changed = base;
        changed.endpoint = Endpoint::new("https://other.test/v1/chat/completions").expect("endpoint");
        assert_ne!(changed.binding().effective_config_sha256, base_digest);
    }

    fn local_server(response: Vec<u8>, hold: Duration) -> (String, std::thread::JoinHandle<()>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("listener");
        let address = listener.local_addr().expect("addr");
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut request = Vec::new();
            let mut buffer = [0u8; 1024];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = stream.read(&mut buffer).unwrap_or(0);
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
            }
            stream.write_all(&response).expect("write response");
            std::thread::sleep(hold);
        });
        (format!("http://{address}/v1/chat/completions"), handle)
    }

    #[tokio::test]
    async fn chunked_response_without_content_length_is_bounded() {
        let chunks = b"hello";
        let response = format!(
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n\
             5\r\nhello\r\n5\r\nworld\r\n0\r\n\r\n"
        )
        .into_bytes();
        assert!(chunks.len() == 5);
        let (endpoint, server) = local_server(response, Duration::from_millis(10));
        let request = TransportRequest {
            endpoint,
            credential: None,
            headers: Vec::new(),
            body: b"{}".to_vec(),
            timeout: Duration::from_secs(2),
            max_response_bytes: 8,
        };
        let error = HttpTransport
            .execute(request, None)
            .await
            .expect_err("oversized chunked body");
        assert_eq!(error, TransportError::TooLarge);
        server.join().expect("server");
    }

    #[tokio::test]
    async fn body_read_timeout_is_classified_as_timeout() {
        let response = b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n{}".to_vec();
        let (endpoint, server) = local_server(response, Duration::from_millis(300));
        let request = TransportRequest {
            endpoint,
            credential: None,
            headers: Vec::new(),
            body: b"{}".to_vec(),
            timeout: Duration::from_millis(50),
            max_response_bytes: 1024,
        };
        let error = HttpTransport
            .execute(request, None)
            .await
            .expect_err("timeout");
        assert_eq!(error, TransportError::Timeout);
        server.join().expect("server");
    }

    #[tokio::test]
    async fn body_read_cancellation_is_classified_as_cancelled() {
        let response = b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n{}".to_vec();
        let (endpoint, server) = local_server(response, Duration::from_millis(300));
        let request = TransportRequest {
            endpoint,
            credential: None,
            headers: Vec::new(),
            body: b"{}".to_vec(),
            timeout: Duration::from_secs(2),
            max_response_bytes: 1024,
        };
        let token = CancellationToken::new();
        let clone = token.clone();
        let task = tokio::spawn(async move {
            HttpTransport.execute(request, Some(&clone)).await
        });
        tokio::time::sleep(Duration::from_millis(30)).await;
        token.cancel();
        let error = task.await.expect("task").expect_err("cancelled");
        assert_eq!(error, TransportError::Cancelled);
        server.join().expect("server");
    }

    #[tokio::test]
    async fn missing_credential_fails_at_transport_boundary() {
        let request = TransportRequest {
            endpoint: "http://127.0.0.1:9/v1/chat/completions".into(),
            credential: Some(
                CredentialRef::new("HARNESS2_MISSING_CREDENTIAL_TEST").expect("reference"),
            ),
            headers: Vec::new(),
            body: b"{}".to_vec(),
            timeout: Duration::from_millis(10),
            max_response_bytes: 1024,
        };
        let error = HttpTransport.execute(request, None).await.expect_err("missing credential");
        assert!(matches!(error, TransportError::Credential(_)));
    }

    #[test]
    fn usage_kinds_are_distinct() {
        assert_ne!(
            Usage::Actual {
                input_tokens: 1,
                output_tokens: 2
            },
            Usage::Estimated {
                input_tokens: 1,
                output_tokens: 2
            }
        );
        assert_ne!(Usage::Unknown, Usage::Estimated { input_tokens: 0, output_tokens: 0 });
        assert_eq!(Usage::default(), Usage::Unknown);
    }

    #[tokio::test]
    async fn request_bounds_reject_before_transport() {
        let transport = MockTransport::with_response(Ok(success("{}", true)));
        let mut config = model_config(ProviderKind::DeepSeek);
        config.max_request_bytes = 16;
        let client = ProviderClient::new(config, Arc::clone(&transport)).expect("client");
        assert!(matches!(
            client.complete("a long system prompt", "a long user prompt", None).await,
            Err(ProviderError::RequestTooLarge { .. })
        ));
        assert_eq!(transport.calls.load(Ordering::SeqCst), 0);

        let transport = MockTransport::with_response(Ok(success("{}", true)));
        let mut config = model_config(ProviderKind::DeepSeek);
        config.capabilities.context_window = 16;
        config.max_output_tokens = 8;
        let client = ProviderClient::new(config, Arc::clone(&transport)).expect("client");
        assert!(matches!(
            client.complete(&"x".repeat(200), "", None).await,
            Err(ProviderError::ContextExceeded { .. })
        ));
        assert_eq!(transport.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn credential_reference_is_passed_only_to_transport() {
        std::env::set_var("HARNESS2_TEST_KEY", SYNTHETIC_CREDENTIAL);
        let transport = MockTransport::with_response(Ok(success("hello", false)));
        let config = model_config(ProviderKind::DeepSeek);
        let binding = config.binding();
        let client = ProviderClient::new(config, Arc::clone(&transport)).expect("client");
        client.complete("s", "u", None).await.expect("completion");
        let requests = transport.requests();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].credential.is_some());
        let text = format!("{:?}", binding);
        assert!(!text.contains(SYNTHETIC_CREDENTIAL));
        assert!(!text.contains("HARNESS2_TEST_KEY"));
    }

    #[tokio::test]
    async fn http_error_does_not_retry_or_fallback() {
        let transport = MockTransport::with_response(Ok(TransportResponse {
            status: 500,
            body: format!("server exploded {SYNTHETIC_SECRET}").into_bytes(),
        }));
        let client = ProviderClient::new(model_config(ProviderKind::OpenAi), Arc::clone(&transport))
            .expect("client");
        let error = client.complete("s", "u", None).await.expect_err("error");
        match error {
            ProviderError::Status { status, message } => {
                assert_eq!(status, 500);
                assert!(!message.contains(SYNTHETIC_SECRET));
            }
            other => panic!("unexpected: {other}"),
        }
        assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn diagnostic_truncation_is_utf8_safe() {
        let text = redact(format!("{SYNTHETIC_SECRET}-{}", "é".repeat(600)), Some(SYNTHETIC_SECRET));
        assert!(!text.contains(SYNTHETIC_SECRET));
        assert!(text.len() <= 503);
        assert!(text.is_char_boundary(text.len()));
        assert!(text.ends_with('…'));
    }

    #[test]
    fn transport_redaction_removes_secret_and_bounds_text() {
        let text = redact(
            format!("failed with {SYNTHETIC_SECRET} {}", "x".repeat(1000)),
            Some(SYNTHETIC_SECRET),
        );
        assert!(!text.contains("sk-secret"));
        assert!(text.contains("[REDACTED]"));
        assert!(text.len() <= 504);
    }

    #[test]
    fn binding_contains_no_credential_and_rejects_endpoint_secrets() {
        std::env::set_var("HARNESS2_TEST_KEY", SYNTHETIC_CREDENTIAL);
        let config = model_config(ProviderKind::DeepSeek);
        let binding = config.binding();
        let text = serde_json::to_string(&binding).expect("binding");
        assert!(!text.contains(SYNTHETIC_CREDENTIAL));
        assert!(!text.contains("authorization"));
        assert!(!text.contains("HARNESS2_TEST_KEY"));
        // Synthetic userinfo used only to verify credential-bearing endpoints are rejected.
        assert!(Endpoint::new("https://synthetic-user:synthetic-pass@example.test/v1").is_err());
        assert!(Endpoint::new("https://example.test/v1?api_key=synthetic-value").is_err());
    }
}
