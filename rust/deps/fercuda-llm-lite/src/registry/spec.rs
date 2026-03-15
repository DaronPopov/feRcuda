use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub name: String,
    pub family: ModelFamily,
    pub source: ModelSource,
    pub format: WeightFormat,
    pub chat_template: ChatTemplate,
    pub context_length: usize,
    #[serde(default)]
    pub tokenizer: TokenizerSource,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub size: String,
    #[serde(default)]
    pub requires_auth: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFamily {
    Llama,
    Phi,
    Gemma,
}

impl ModelFamily {
    pub fn gguf_arch(&self) -> &'static [&'static str] {
        match self {
            Self::Llama => &["llama", "mistral", "qwen", "qwen2", "qwen3"],
            Self::Phi => &["phi", "phi2", "phi3"],
            Self::Gemma => &["gemma", "gemma2"],
        }
    }

    pub fn from_gguf_arch(arch: &str) -> Option<Self> {
        let arch_lower = arch.to_lowercase();
        for family in [Self::Llama, Self::Phi, Self::Gemma] {
            if family.gguf_arch().iter().any(|a| arch_lower.contains(a)) {
                return Some(family);
            }
        }
        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelSource {
    HuggingFace {
        repo: String,
        #[serde(default)]
        file: Option<String>,
        #[serde(default)]
        revision: Option<String>,
    },
    Local {
        path: PathBuf,
    },
    Url {
        url: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum WeightFormat {
    #[default]
    GGUF,
    SafeTensors,
    PyTorch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ChatTemplate {
    #[default]
    Mistral,
    Llama3,
    ChatML,
    Phi3,
    Gemma,
    Raw,
}

impl ChatTemplate {
    pub fn format(&self, messages: &[ChatMessage]) -> String {
        match self {
            Self::Mistral => {
                let mut result = String::new();
                for msg in messages {
                    match msg.role {
                        Role::System | Role::User => {
                            result.push_str(&format!("[INST] {} [/INST]", msg.content));
                        }
                        Role::Assistant => result.push_str(&msg.content),
                    }
                }
                result
            }
            Self::Llama3 => {
                let mut result = "<|begin_of_text|>".to_string();
                for msg in messages {
                    let role = match msg.role {
                        Role::System => "system",
                        Role::User => "user",
                        Role::Assistant => "assistant",
                    };
                    result.push_str(&format!(
                        "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                        role, msg.content
                    ));
                }
                result.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                result
            }
            Self::ChatML => {
                let mut result = String::new();
                for msg in messages {
                    let role = match msg.role {
                        Role::System => "system",
                        Role::User => "user",
                        Role::Assistant => "assistant",
                    };
                    result.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, msg.content));
                }
                result.push_str("<|im_start|>assistant\n");
                result
            }
            Self::Phi3 => {
                let mut result = String::new();
                for msg in messages {
                    let role = match msg.role {
                        Role::System => "system",
                        Role::User => "user",
                        Role::Assistant => "assistant",
                    };
                    result.push_str(&format!("<|{}|>\n{}<|end|>\n", role, msg.content));
                }
                result.push_str("<|assistant|>\n");
                result
            }
            Self::Gemma => {
                let mut result = String::new();
                for msg in messages {
                    let role = match msg.role {
                        Role::Assistant => "model",
                        _ => "user",
                    };
                    result.push_str(&format!(
                        "<start_of_turn>{}\n{}<end_of_turn>\n",
                        role, msg.content
                    ));
                }
                result.push_str("<start_of_turn>model\n");
                result
            }
            Self::Raw => messages.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TokenizerSource {
    #[default]
    SameAsModel,
    HuggingFace { repo: String },
    Local { path: PathBuf },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub description: String,
    pub size: String,
    pub family: ModelFamily,
    pub requires_auth: bool,
}

impl From<&ModelSpec> for ModelInfo {
    fn from(spec: &ModelSpec) -> Self {
        Self {
            name: spec.name.clone(),
            description: spec.description.clone(),
            size: spec.size.clone(),
            family: spec.family,
            requires_auth: spec.requires_auth,
        }
    }
}
