use anyhow::{Context, Result};
use candle_core::{quantized::gguf_file, Device, Tensor as CandleTensor};
use clap::{Parser, Subcommand};
use fercuda_llm_lite::{
    ChatMessage, DownloadedModel, GenerationConfig, InferenceModel, ModelLoader,
    StreamingInference, Tokenizer,
};
use ptx_os::prelude::*;
use ptx_os::vfs::OpenFlags;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "fercuda-llm-app")]
#[command(about = "LLM application-layer tooling on top of feRcuda PTX-OS runtime")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Catalog {
        #[arg(long, default_value = "./models")]
        cache_dir: String,
    },
    Download {
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "./models")]
        cache_dir: String,
        #[arg(long)]
        hf_token: Option<String>,
    },
    RuntimeSmoke {
        #[arg(long, default_value_t = 0)]
        device: i32,
        #[arg(long, default_value_t = 0.90)]
        pool_fraction: f32,
        #[arg(long, default_value_t = 64)]
        alloc_mb: usize,
    },
    PrepareSession {
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "./models")]
        cache_dir: String,
        #[arg(long)]
        hf_token: Option<String>,
        #[arg(long, default_value_t = 0)]
        device: i32,
        #[arg(long, default_value_t = 0.90)]
        pool_fraction: f32,
        #[arg(long, default_value_t = 1024)]
        kv_cache_tokens: usize,
        #[arg(long, default_value_t = 4096)]
        hidden_size: usize,
        #[arg(long, default_value = "/llm")]
        vfs_root: String,
    },
    MistralExample {
        #[arg(long, default_value = "mistral-7b-q4")]
        model: String,
        #[arg(long, default_value = "./models")]
        cache_dir: String,
        #[arg(long)]
        hf_token: Option<String>,
        #[arg(long, default_value = "Write one short paragraph about GPU allocators.")]
        prompt: String,
        #[arg(long)]
        system: Option<String>,
        #[arg(long, default_value_t = 0)]
        device: usize,
        #[arg(long)]
        cpu: bool,
        #[arg(long, default_value_t = 96)]
        max_tokens: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Catalog { cache_dir } => cmd_catalog(&cache_dir),
        Command::Download {
            model,
            cache_dir,
            hf_token,
        } => cmd_download(&model, &cache_dir, hf_token),
        Command::RuntimeSmoke {
            device,
            pool_fraction,
            alloc_mb,
        } => cmd_runtime_smoke(device, pool_fraction, alloc_mb),
        Command::PrepareSession {
            model,
            cache_dir,
            hf_token,
            device,
            pool_fraction,
            kv_cache_tokens,
            hidden_size,
            vfs_root,
        } => cmd_prepare_session(
            &model,
            &cache_dir,
            hf_token,
            device,
            pool_fraction,
            kv_cache_tokens,
            hidden_size,
            &vfs_root,
        ),
        Command::MistralExample {
            model,
            cache_dir,
            hf_token,
            prompt,
            system,
            device,
            cpu,
            max_tokens,
        } => cmd_mistral_example(
            &model,
            &cache_dir,
            hf_token,
            &prompt,
            system.as_deref(),
            device,
            cpu,
            max_tokens,
        ),
    }
}

fn cmd_catalog(cache_dir: &str) -> Result<()> {
    let loader = ModelLoader::new(cache_dir);
    println!("Built-in model catalog:");
    for model in loader.list_models() {
        println!(
            "- {:16} size={} auth={} desc={}",
            model.name, model.size, model.requires_auth, model.description
        );
    }
    Ok(())
}

fn cmd_download(model: &str, cache_dir: &str, hf_token: Option<String>) -> Result<()> {
    let downloaded = loader(cache_dir, hf_token)
        .download(model)
        .with_context(|| format!("failed to download model {}", model))?;
    print_downloaded(&downloaded);
    Ok(())
}

fn cmd_runtime_smoke(device: i32, pool_fraction: f32, alloc_mb: usize) -> Result<()> {
    let runtime = init_runtime(device, pool_fraction)?;
    let tensor = Tensor::zeros(
        &runtime,
        &[(alloc_mb * 1024 * 1024 / 4)],
        ptx_os::tensor::DType::Float32,
    )?;
    let stats = runtime.stats();
    let pool = runtime.pool_stats();

    println!("Runtime smoke ok:");
    println!("  device={}", runtime.device_id());
    println!("  tensor_bytes={}", tensor.size_bytes());
    println!("  vram_used_mb={:.2}", stats.vram_used as f64 / 1024.0 / 1024.0);
    println!("  pool_utilization={:.2}%", pool.utilization_percent);
    println!("  pool_healthy={}", pool.is_healthy);
    Ok(())
}

fn cmd_prepare_session(
    model: &str,
    cache_dir: &str,
    hf_token: Option<String>,
    device: i32,
    pool_fraction: f32,
    kv_cache_tokens: usize,
    hidden_size: usize,
    vfs_root: &str,
) -> Result<()> {
    let downloaded = loader(cache_dir, hf_token)
        .download(model)
        .with_context(|| format!("failed to download model {}", model))?;
    let runtime = init_runtime(device, pool_fraction)?;
    let vfs = ptx_os::VirtualFs::new(&runtime).context("failed to initialize PTX-OS VFS")?;

    ensure_dir(&vfs, vfs_root)?;
    ensure_dir(&vfs, &format!("{}/models", vfs_root))?;
    ensure_dir(&vfs, &format!("{}/sessions", vfs_root))?;
    ensure_dir(&vfs, &format!("{}/sessions/default", vfs_root))?;

    let manifest = serde_json::json!({
        "model_name": downloaded.spec.name,
        "weights_path": downloaded.weights_path,
        "tokenizer_path": downloaded.tokenizer_path,
        "context_length": downloaded.spec.context_length,
        "pool_fraction": pool_fraction,
        "device": device,
    });
    write_vfs_file(
        &vfs,
        &format!("{}/sessions/default/manifest.json", vfs_root),
        manifest.to_string().as_bytes(),
    )?;

    vfs.create_tensor(
        &format!("{}/sessions/default/kv_cache", vfs_root),
        &[kv_cache_tokens as i32, hidden_size as i32],
        ptx_os::tensor::DType::Float16,
    )?;
    vfs.create_tensor(
        &format!("{}/sessions/default/logits", vfs_root),
        &[1, 32000],
        ptx_os::tensor::DType::Float32,
    )?;

    let stats = runtime.stats();
    let pool = runtime.pool_stats();
    println!("Prepared LLM session:");
    print_downloaded(&downloaded);
    println!("  device={}", runtime.device_id());
    println!("  vfs_root={}", vfs_root);
    println!(
        "  kv_cache_shape=[{}, {}]",
        kv_cache_tokens, hidden_size
    );
    println!("  pool_utilization={:.2}%", pool.utilization_percent);
    println!("  vram_used_mb={:.2}", stats.vram_used as f64 / 1024.0 / 1024.0);
    Ok(())
}

fn cmd_mistral_example(
    model: &str,
    cache_dir: &str,
    hf_token: Option<String>,
    prompt: &str,
    system: Option<&str>,
    device: usize,
    cpu: bool,
    max_tokens: usize,
) -> Result<()> {
    let downloaded = loader(cache_dir, hf_token)
        .download(model)
        .with_context(|| format!("failed to download model {}", model))?;
    let tokenizer = Tokenizer::from_dir(&downloaded.tokenizer_path)
        .with_context(|| format!("failed to load tokenizer from {}", downloaded.tokenizer_path.display()))?;

    let formatted_prompt = if tokenizer.has_chat_template() {
        tokenizer.apply_chat_template(
            &build_messages(prompt, system),
            true,
        )?
    } else {
        let mut formatted = String::new();
        if let Some(system) = system {
            formatted.push_str(system);
            formatted.push_str("\n\n");
        }
        formatted.push_str(prompt);
        formatted
    };

    let candle_device = select_candle_device(device, cpu)?;
    let mut model = QuantizedLlama::new(&downloaded.weights_path, candle_device)?;
    let config = GenerationConfig::greedy().with_max_tokens(max_tokens);
    let mut inference = StreamingInference::new(&mut model, &tokenizer, &formatted_prompt, config)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    print_downloaded(&downloaded);
    println!("  backend=quantized-llama(gguf)");
    println!("  prompt={}", prompt);
    println!("  mode={}", if cpu { "cpu" } else { "cuda" });
    println!();

    let mut output = String::new();
    while !inference.is_finished() {
        if let Some(chunk) = inference
            .next()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
        {
            print!("{chunk}");
            output.push_str(&chunk);
        }
    }
    println!();
    println!("[stats] tok/s={:.1}", inference.stats().tokens_per_second());
    if output.is_empty() {
        println!("[note] no decoded text emitted before stop condition");
    }
    Ok(())
}

fn loader(cache_dir: &str, hf_token: Option<String>) -> ModelLoader {
    ModelLoader::new(cache_dir).with_auth(hf_token.or_else(|| std::env::var("HF_TOKEN").ok()))
}

fn init_runtime(device: i32, pool_fraction: f32) -> Result<RegimeRuntimeCore> {
    let regime = RegimeConfig::new()
        .pool_fraction(pool_fraction)
        .alloc_mode(AllocMode::SessionBuffers)
        .quiet();
    RegimeRuntimeCore::with_regime(device, regime).context("failed to initialize PTX-OS runtime")
}

fn ensure_dir(vfs: &ptx_os::VirtualFs, path: &str) -> Result<()> {
    match vfs.mkdir(path, 0o755) {
        Ok(()) => Ok(()),
        Err(_) => Ok(()),
    }
}

fn write_vfs_file(vfs: &ptx_os::VirtualFs, path: &str, data: &[u8]) -> Result<()> {
    let mut file = vfs.open(
        path,
        OpenFlags::new().write().create().truncate(),
    )?;
    let written = file.write(data)?;
    if written != data.len() {
        anyhow::bail!("short write to {}", path);
    }
    Ok(())
}

fn print_downloaded(downloaded: &DownloadedModel) {
    println!("  model={}", downloaded.spec.name);
    println!("  weights={}", downloaded.weights_path.display());
    println!("  tokenizer={}", downloaded.tokenizer_path.display());
    println!("  context_length={}", downloaded.spec.context_length);
}

fn build_messages(prompt: &str, system: Option<&str>) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    if let Some(system) = system {
        messages.push(ChatMessage::system(system));
    }
    messages.push(ChatMessage::user(prompt));
    messages
}

fn select_candle_device(device: usize, force_cpu: bool) -> Result<Device> {
    if force_cpu {
        return Ok(Device::Cpu);
    }
    Device::new_cuda(device).context("failed to create Candle CUDA device")
}

struct QuantizedLlama {
    model: candle_transformers::models::quantized_llama::ModelWeights,
    device: Device,
}

impl QuantizedLlama {
    fn new(gguf_path: &PathBuf, device: Device) -> Result<Self> {
        let mut file = std::fs::File::open(gguf_path)
            .with_context(|| format!("failed to open {}", gguf_path.display()))?;
        let gguf_content = gguf_file::Content::read(&mut file)
            .context("failed to read gguf content")?;
        let model =
            candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
                gguf_content,
                &mut file,
                &device,
            )
            .context("failed to construct quantized llama weights from gguf")?;
        Ok(Self { model, device })
    }
}

impl InferenceModel for QuantizedLlama {
    fn forward(
        &mut self,
        tokens: &[u32],
        pos: usize,
    ) -> Result<CandleTensor, Box<dyn std::error::Error>> {
        if tokens.is_empty() {
            return Ok(CandleTensor::zeros((1, 32000), candle_core::DType::F32, &self.device)?);
        }
        let input = CandleTensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, pos)?;
        Ok(logits.squeeze(0)?)
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<CandleTensor, Box<dyn std::error::Error>> {
        let mut pos = 0;
        for &token in tokens.iter().take(tokens.len().saturating_sub(1)) {
            let input = CandleTensor::new(&[token], &self.device)?.unsqueeze(0)?;
            let _ = self.model.forward(&input, pos)?;
            pos += 1;
        }

        if let Some(&last_token) = tokens.last() {
            let input = CandleTensor::new(&[last_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, pos)?;
            Ok(logits.squeeze(0)?)
        } else {
            Ok(CandleTensor::zeros((1, 32000), candle_core::DType::F32, &self.device)?)
        }
    }

    fn clear_cache(&mut self) {}
}
