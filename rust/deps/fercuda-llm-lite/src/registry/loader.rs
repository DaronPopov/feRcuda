use super::catalog::Catalog;
use super::spec::*;
use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

#[derive(Debug, Clone)]
pub struct DownloadedModel {
    pub spec: ModelSpec,
    pub weights_path: PathBuf,
    pub tokenizer_path: PathBuf,
}

impl DownloadedModel {
    pub fn name(&self) -> &str {
        &self.spec.name
    }
}

pub struct ModelLoader {
    catalog: Catalog,
    #[allow(dead_code)]
    cache_dir: PathBuf,
    auth_token: Option<String>,
}

impl ModelLoader {
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            catalog: Catalog::new(),
            cache_dir: cache_dir.into(),
            auth_token: None,
        }
    }

    pub fn with_auth(mut self, token: Option<String>) -> Self {
        self.auth_token = token;
        self
    }

    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    pub fn list_models(&self) -> Vec<ModelInfo> {
        self.catalog.list().iter().map(|s| ModelInfo::from(*s)).collect()
    }

    pub fn search(&self, query: &str) -> Vec<ModelInfo> {
        self.catalog.search(query).iter().map(|s| ModelInfo::from(*s)).collect()
    }

    pub fn get_spec(&self, name: &str) -> Option<&ModelSpec> {
        self.catalog.get(name)
    }

    pub fn download(&self, name: &str) -> Result<DownloadedModel> {
        let spec = self
            .catalog
            .get(name)
            .ok_or_else(|| anyhow!("Unknown model: {}", name))?;
        self.download_spec(spec)
    }

    pub fn download_spec(&self, spec: &ModelSpec) -> Result<DownloadedModel> {
        info!("Downloading model: {} ({:?})", spec.name, spec.family);
        if spec.requires_auth && self.auth_token.is_none() {
            warn!("Model {} requires authentication. Set HF_TOKEN.", spec.name);
        }

        let weights_path = self.download_weights(spec)?;
        let tokenizer_path = self.download_tokenizer(spec)?;
        Ok(DownloadedModel {
            spec: spec.clone(),
            weights_path,
            tokenizer_path,
        })
    }

    fn download_weights(&self, spec: &ModelSpec) -> Result<PathBuf> {
        match &spec.source {
            ModelSource::HuggingFace { repo, file, revision } => {
                let api = self.create_hf_api()?;
                let repo_api = match revision {
                    Some(rev) => api.repo(hf_hub::Repo::with_revision(
                        repo.clone(),
                        hf_hub::RepoType::Model,
                        rev.clone(),
                    )),
                    None => api.model(repo.clone()),
                };
                let filename = file
                    .as_ref()
                    .ok_or_else(|| anyhow!("No filename specified for HuggingFace model"))?;
                repo_api
                    .get(filename)
                    .with_context(|| format!("Failed to download {} from {}", filename, repo))
            }
            ModelSource::Local { path } => {
                if path.exists() {
                    Ok(path.clone())
                } else {
                    Err(anyhow!("Local model not found: {}", path.display()))
                }
            }
            ModelSource::Url { url } => Err(anyhow!("URL download not implemented: {}", url)),
        }
    }

    fn download_tokenizer(&self, spec: &ModelSpec) -> Result<PathBuf> {
        match &spec.tokenizer {
            TokenizerSource::SameAsModel => match &spec.source {
                ModelSource::HuggingFace { repo, .. } => self.download_tokenizer_repo(repo),
                ModelSource::Local { path } => Ok(path.parent().unwrap_or(path).to_path_buf()),
                ModelSource::Url { .. } => Err(anyhow!("Cannot derive tokenizer from URL source")),
            },
            TokenizerSource::HuggingFace { repo } => self.download_tokenizer_repo(repo),
            TokenizerSource::Local { path } => Ok(path.clone()),
        }
    }

    fn download_tokenizer_repo(&self, repo: &str) -> Result<PathBuf> {
        let api = self.create_hf_api()?;
        let repo_api = api.model(repo.to_string());
        let tokenizer_path = repo_api
            .get("tokenizer.json")
            .with_context(|| format!("Failed to download tokenizer.json from {}", repo))?;
        let _ = repo_api.get("tokenizer_config.json");
        Ok(tokenizer_path.parent().unwrap_or(&tokenizer_path).to_path_buf())
    }

    fn create_hf_api(&self) -> Result<hf_hub::api::sync::Api> {
        let mut builder = hf_hub::api::sync::ApiBuilder::new();
        if let Some(ref token) = self.auth_token {
            builder = builder.with_token(Some(token.clone()));
            debug!("Using HuggingFace authentication token");
        }
        builder.build().context("Failed to create HuggingFace API client")
    }

    pub fn load_gguf_auto(&self, path: &Path) -> Result<DownloadedModel> {
        let metadata = read_gguf_metadata(path)?;
        let arch = metadata
            .get("general.architecture")
            .ok_or_else(|| anyhow!("GGUF file missing architecture metadata"))?;
        let family = ModelFamily::from_gguf_arch(arch)
            .ok_or_else(|| anyhow!("Unknown architecture: {}", arch))?;
        let name = metadata
            .get("general.name")
            .cloned()
            .unwrap_or_else(|| path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string());
        let context_length = metadata
            .get("llama.context_length")
            .or_else(|| metadata.get("phi.context_length"))
            .or_else(|| metadata.get("gemma.context_length"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(4096);

        let spec = ModelSpec {
            name,
            family,
            source: ModelSource::Local { path: path.to_path_buf() },
            format: WeightFormat::GGUF,
            chat_template: detect_chat_template(&metadata, family),
            context_length,
            tokenizer: TokenizerSource::SameAsModel,
            description: format!("Auto-detected {:?} model", family),
            size: detect_size(&metadata),
            requires_auth: false,
        };

        Ok(DownloadedModel {
            spec,
            weights_path: path.to_path_buf(),
            tokenizer_path: path.parent().unwrap_or(path).to_path_buf(),
        })
    }
}

fn read_gguf_metadata(path: &Path) -> Result<HashMap<String, String>> {
    use std::fs::File;
    use std::io::{BufReader, Read};

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(anyhow!("Invalid GGUF file: bad magic number"));
    }

    let mut version = [0u8; 4];
    reader.read_exact(&mut version)?;
    let _version = u32::from_le_bytes(version);

    let mut tensor_count = [0u8; 8];
    reader.read_exact(&mut tensor_count)?;
    let mut metadata_count = [0u8; 8];
    reader.read_exact(&mut metadata_count)?;
    let metadata_count = u64::from_le_bytes(metadata_count);

    let mut metadata = HashMap::new();
    for _ in 0..metadata_count.min(100) {
        let key = match read_gguf_string(&mut reader) {
            Ok(k) => k,
            Err(_) => break,
        };

        let mut value_type = [0u8; 4];
        if reader.read_exact(&mut value_type).is_err() {
            break;
        }
        let value_type = u32::from_le_bytes(value_type);
        let value = match value_type {
            8 => read_gguf_string(&mut reader).unwrap_or_default(),
            4 => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf).ok();
                u32::from_le_bytes(buf).to_string()
            }
            10 => {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf).ok();
                u64::from_le_bytes(buf).to_string()
            }
            _ => continue,
        };
        if !key.is_empty() {
            metadata.insert(key, value);
        }
    }

    Ok(metadata)
}

fn read_gguf_string<R: std::io::Read>(reader: &mut R) -> Result<String> {
    let mut len_buf = [0u8; 8];
    reader.read_exact(&mut len_buf)?;
    let len = u64::from_le_bytes(len_buf) as usize;
    if len > 1024 * 1024 {
        return Err(anyhow!("String too long: {} bytes", len));
    }
    let mut string_buf = vec![0u8; len];
    reader.read_exact(&mut string_buf)?;
    String::from_utf8(string_buf).context("Invalid UTF-8 in GGUF string")
}

fn detect_chat_template(metadata: &HashMap<String, String>, family: ModelFamily) -> ChatTemplate {
    if let Some(template) = metadata.get("tokenizer.chat_template") {
        if template.contains("[INST]") {
            return ChatTemplate::Mistral;
        }
        if template.contains("<|im_start|>") {
            return ChatTemplate::ChatML;
        }
        if template.contains("<start_of_turn>") {
            return ChatTemplate::Gemma;
        }
    }

    match family {
        ModelFamily::Llama => ChatTemplate::Mistral,
        ModelFamily::Phi => ChatTemplate::Phi3,
        ModelFamily::Gemma => ChatTemplate::Gemma,
    }
}

fn detect_size(metadata: &HashMap<String, String>) -> String {
    if let Some(params) = metadata.get("general.parameter_count") {
        if let Ok(value) = params.parse::<f64>() {
            return format!("{:.1}B", value / 1_000_000_000.0);
        }
    }
    "unknown".into()
}
