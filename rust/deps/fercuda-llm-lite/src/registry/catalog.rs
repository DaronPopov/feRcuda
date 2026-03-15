use super::spec::*;
use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct Catalog {
    models: HashMap<String, ModelSpec>,
}

impl Catalog {
    pub fn new() -> Self {
        let mut catalog = Self::default();
        catalog.register_builtins();
        catalog
    }

    fn register_builtins(&mut self) {
        self.register(ModelSpec {
            name: "mistral-7b-q4".into(),
            family: ModelFamily::Llama,
            source: ModelSource::HuggingFace {
                repo: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF".into(),
                file: Some("mistral-7b-instruct-v0.2.Q4_K_M.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::Mistral,
            context_length: 32768,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "mistralai/Mistral-7B-Instruct-v0.2".into(),
            },
            description: "Mistral 7B Instruct v0.2 (4-bit quantized)".into(),
            size: "7B".into(),
            requires_auth: false,
        });

        self.register(ModelSpec {
            name: "qwen2-0.5b-q4".into(),
            family: ModelFamily::Llama,
            source: ModelSource::HuggingFace {
                repo: "Qwen/Qwen2-0.5B-Instruct-GGUF".into(),
                file: Some("qwen2-0_5b-instruct-q4_k_m.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::ChatML,
            context_length: 32768,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "Qwen/Qwen2-0.5B-Instruct".into(),
            },
            description: "Qwen2 0.5B Instruct (4-bit quantized)".into(),
            size: "0.5B".into(),
            requires_auth: false,
        });

        self.register(ModelSpec {
            name: "qwen3-8b-q4".into(),
            family: ModelFamily::Llama,
            source: ModelSource::HuggingFace {
                repo: "Qwen/Qwen3-8B-GGUF".into(),
                file: Some("Qwen3-8B-Q4_K_M.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::ChatML,
            context_length: 32768,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "Qwen/Qwen3-8B".into(),
            },
            description: "Qwen3 8B Instruct (4-bit quantized)".into(),
            size: "8B".into(),
            requires_auth: false,
        });
    }

    pub fn register(&mut self, spec: ModelSpec) {
        self.models.insert(spec.name.clone(), spec);
    }

    pub fn list(&self) -> Vec<&ModelSpec> {
        let mut specs: Vec<_> = self.models.values().collect();
        specs.sort_by(|a, b| a.name.cmp(&b.name));
        specs
    }

    pub fn search(&self, query: &str) -> Vec<&ModelSpec> {
        let query_lower = query.to_lowercase();
        self.list()
            .into_iter()
            .filter(|spec| {
                spec.name.to_lowercase().contains(&query_lower)
                    || spec.description.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    pub fn get(&self, name: &str) -> Option<&ModelSpec> {
        self.models.get(name)
    }
}
