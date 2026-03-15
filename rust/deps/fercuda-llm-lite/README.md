# fercuda-llm-lite

Minimal Rust LLM utility layer extracted from `ferrite-llm` and embedded into `feRcuda`.

Included:
- tokenizer loading and chat-template rendering
- sampling and streaming inference scaffolding
- a small built-in GGUF/HuggingFace model catalog
- HF-backed model/tokenizer download helpers

Deliberately excluded:
- `ferrite-llm` CLI binaries
- WASM guest/host runtime
- embedded `ferrite-os` tree
- direct TLSF allocator wiring, because `feRcuda`'s Rust surface is organized differently

Build:

```bash
cargo check --manifest-path rust/deps/fercuda-llm-lite/Cargo.toml
```
