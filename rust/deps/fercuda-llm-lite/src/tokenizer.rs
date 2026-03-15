use llm_tokenizer::{
    chat_template::{load_chat_template_from_config, ChatTemplateParams, ChatTemplateProcessor},
    hub::download_tokenizer_from_hf,
    DecodeStream, Encoding as LlmEncoding, Tokenizer as LlmTokenizer,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Encoding {
    pub ids: Vec<u32>,
}

impl Encoding {
    fn from_llm(encoding: &LlmEncoding) -> Self {
        Self {
            ids: encoding.token_ids().to_vec(),
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

pub struct Tokenizer {
    inner: Arc<LlmTokenizer>,
    vocab_size: usize,
    model_dir: Option<PathBuf>,
    chat_template: Option<ChatTemplateProcessor>,
}

impl Tokenizer {
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let dir = dir.as_ref();
        let tokenizer_path = dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(TokenizerError::LoadFailed(format!(
                "tokenizer.json not found in {}",
                dir.display()
            )));
        }

        let tokenizer = LlmTokenizer::from_file(tokenizer_path.to_string_lossy().as_ref())
            .map_err(|e| TokenizerError::LoadFailed(e.to_string()))?;
        let vocab_size = tokenizer.vocab_size();

        let config_path = dir.join("tokenizer_config.json");
        let chat_template = if config_path.exists() {
            load_chat_template_from_config(config_path.to_string_lossy().as_ref())
                .ok()
                .flatten()
                .map(ChatTemplateProcessor::new)
                .transpose()
                .map_err(|e| TokenizerError::TemplateFailed(e.to_string()))?
        } else {
            None
        };

        Ok(Self {
            inner: Arc::new(tokenizer),
            vocab_size,
            model_dir: Some(dir.to_path_buf()),
            chat_template,
        })
    }

    pub async fn from_pretrained(model_id: &str) -> Result<Self, TokenizerError> {
        let model_dir = download_tokenizer_from_hf(model_id)
            .await
            .map_err(|e| TokenizerError::LoadFailed(e.to_string()))?;
        Self::from_dir(&model_dir)
    }

    pub fn encode(&self, text: &str) -> Result<Encoding, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| TokenizerError::EncodeFailed(e.to_string()))?;
        Ok(Encoding::from_llm(&encoding))
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| TokenizerError::DecodeFailed(e.to_string()))
    }

    pub fn decode_stream(&self, prompt_tokens: &[u32], skip_special: bool) -> StreamDecoder {
        StreamDecoder {
            stream: self.inner.decode_stream(prompt_tokens, skip_special),
        }
    }

    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        let processor = self
            .chat_template
            .as_ref()
            .ok_or_else(|| TokenizerError::TemplateFailed("No chat template available".to_string()))?;

        let json_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role.as_str(),
                    "content": m.content,
                })
            })
            .collect();

        let special = self.inner.get_special_tokens();
        let mut kwargs = std::collections::HashMap::new();
        if let Some(ref bos) = special.bos_token {
            kwargs.insert("bos_token".to_string(), serde_json::json!(bos));
        }
        if let Some(ref eos) = special.eos_token {
            kwargs.insert("eos_token".to_string(), serde_json::json!(eos));
        }

        let params = ChatTemplateParams {
            add_generation_prompt,
            template_kwargs: Some(&kwargs),
            ..Default::default()
        };

        processor
            .apply_chat_template(&json_messages, params)
            .map_err(|e| TokenizerError::TemplateFailed(e.to_string()))
    }

    pub fn encode_chat(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<Encoding, TokenizerError> {
        let text = self.apply_chat_template(messages, add_generation_prompt)?;
        self.encode(&text)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.inner
            .get_special_tokens()
            .eos_token
            .as_ref()
            .and_then(|t| self.inner.token_to_id(t))
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.inner
            .get_special_tokens()
            .bos_token
            .as_ref()
            .and_then(|t| self.inner.token_to_id(t))
    }

    pub fn has_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    pub fn model_dir(&self) -> Option<&Path> {
        self.model_dir.as_deref()
    }
}

pub struct StreamDecoder {
    stream: DecodeStream,
}

impl StreamDecoder {
    pub fn step(&mut self, token_id: u32) -> Result<Option<String>, TokenizerError> {
        self.stream
            .step(token_id)
            .map_err(|e| TokenizerError::DecodeFailed(e.to_string()))
    }

    pub fn flush(&mut self) -> Result<Option<String>, TokenizerError> {
        self.stream
            .flush()
            .map_err(|e| TokenizerError::DecodeFailed(e.to_string()))
    }
}

#[derive(Debug, Clone)]
pub enum TokenizerError {
    LoadFailed(String),
    EncodeFailed(String),
    DecodeFailed(String),
    TemplateFailed(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::LoadFailed(e) => write!(f, "Load failed: {}", e),
            TokenizerError::EncodeFailed(e) => write!(f, "Encode failed: {}", e),
            TokenizerError::DecodeFailed(e) => write!(f, "Decode failed: {}", e),
            TokenizerError::TemplateFailed(e) => write!(f, "Template failed: {}", e),
        }
    }
}

impl std::error::Error for TokenizerError {}
