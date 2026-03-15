use fercuda_llm_lite::{ModelConfig, ModelLoader};

fn main() {
    let loader = ModelLoader::new("./models");

    println!("Integrated models:");
    for model in loader.list_models() {
        println!(
            "- {} [{}] auth={} ",
            model.name, model.size, model.requires_auth
        );
    }

    if let Some(model) = ModelConfig::find("qwen3") {
        println!(
            "Estimated VRAM for {}: {:.1} GB fp16 / {:.1} GB q4",
            model.name,
            model.estimated_vram_gb(false),
            model.estimated_vram_gb(true)
        );
    }
}
