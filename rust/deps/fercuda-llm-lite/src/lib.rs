//! Lightweight LLM utilities extracted from ferrite-llm and embedded into feRcuda.
//!
//! This crate intentionally keeps only the portable Rust layer:
//! - tokenizer loading and chat-template handling
//! - generation/session scaffolding
//! - sampling helpers
//! - a small model catalog and HF-backed loader
//!
//! It does not embed the ferrite-llm CLI, WASM runtime, or its separate ferrite-os tree.

pub mod generation;
pub mod hooks;
pub mod models;
pub mod registry;
pub mod sampling;
pub mod tokenizer;

pub use generation::{
    GenerationConfig, GenerationStats, InferenceModel, SpeculativeInference, StopCondition,
    StreamingInference,
};
pub use hooks::{LogitsCandidate, LogitsHook};
pub use models::{ChatFormat, ModelConfig, ModelFamily};
pub use registry::{
    Catalog, ChatMessage as RegistryChatMessage, ChatTemplate, DownloadedModel, ModelInfo,
    ModelLoader, ModelSource, ModelSpec, Role, TokenizerSource, WeightFormat,
};
pub use sampling::{Sampler, SamplerConfig};
pub use tokenizer::{ChatMessage, ChatRole, Encoding, StreamDecoder, Tokenizer, TokenizerError};
