use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFamily {
    Llama,
    Mistral,
    Qwen,
    Gemma,
    Phi,
}

impl ModelFamily {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "llama" | "tinyllama" => Some(Self::Llama),
            "mistral" => Some(Self::Mistral),
            "qwen" | "qwen2" | "qwen3" => Some(Self::Qwen),
            "gemma" => Some(Self::Gemma),
            "phi" | "phi2" | "phi-2" | "phi3" => Some(Self::Phi),
            _ => None,
        }
    }

    pub fn chat_format(&self) -> ChatFormat {
        match self {
            Self::Llama => ChatFormat::Llama,
            Self::Mistral => ChatFormat::Mistral,
            Self::Qwen => ChatFormat::ChatML,
            Self::Gemma => ChatFormat::Gemma,
            Self::Phi => ChatFormat::Phi,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ChatFormat {
    Llama,
    Mistral,
    ChatML,
    Gemma,
    Phi,
}

impl ChatFormat {
    pub fn format_simple(&self, user_input: &str) -> String {
        match self {
            ChatFormat::Llama | ChatFormat::Mistral => format!("[INST] {} [/INST]", user_input),
            ChatFormat::ChatML => format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                user_input
            ),
            ChatFormat::Gemma => format!(
                "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
                user_input
            ),
            ChatFormat::Phi => format!("Instruct: {}\nOutput:", user_input),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_id: String,
    pub family: ModelFamily,
    pub name: String,
    pub parameters: String,
    pub context_length: usize,
    pub quantized: bool,
    pub requires_auth: bool,
}

impl ModelConfig {
    pub fn registry() -> Vec<ModelConfig> {
        vec![
            ModelConfig {
                model_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".into(),
                family: ModelFamily::Llama,
                name: "TinyLlama 1.1B".into(),
                parameters: "1.1B".into(),
                context_length: 2048,
                quantized: false,
                requires_auth: false,
            },
            ModelConfig {
                model_id: "mistralai/Mistral-7B-Instruct-v0.2".into(),
                family: ModelFamily::Mistral,
                name: "Mistral 7B Instruct".into(),
                parameters: "7B".into(),
                context_length: 32768,
                quantized: false,
                requires_auth: false,
            },
            ModelConfig {
                model_id: "Qwen/Qwen3-8B".into(),
                family: ModelFamily::Qwen,
                name: "Qwen3 8B".into(),
                parameters: "8B".into(),
                context_length: 32768,
                quantized: false,
                requires_auth: false,
            },
        ]
    }

    pub fn find(query: &str) -> Option<ModelConfig> {
        let query_lower = query.to_lowercase();
        Self::registry().into_iter().find(|m| {
            m.model_id.to_lowercase().contains(&query_lower)
                || m.name.to_lowercase().contains(&query_lower)
        })
    }

    pub fn estimated_vram_gb(&self, quantized: bool) -> f32 {
        let params_str = self.parameters.to_lowercase();
        let billions = params_str
            .trim_end_matches(" (4-bit)")
            .trim_end_matches('b')
            .parse::<f32>()
            .unwrap_or(1.0);

        if quantized {
            (billions * 0.55) + 0.5
        } else {
            (billions * 2.0) + 0.5
        }
    }
}
