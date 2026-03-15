# fercuda-llm-app

Application-layer LLM tooling for `feRcuda`.

This crate is intentionally above the low-level runtime:
- model catalog and HuggingFace download logic come from `fercuda-llm-lite`
- runtime/session allocation and GPU-resident VFS layout come from `ptx_os`

It does not implement full token generation yet. It prepares the application path:
- list catalog entries
- download weights/tokenizer
- initialize the existing PTX-OS runtime in this repo
- allocate session tensors and write a session manifest into the GPU VFS

Examples:

```bash
LD_LIBRARY_PATH=/home/daron/feRcuda/build \
  cargo run --manifest-path rust/apps/fercuda-llm-app/Cargo.toml -- catalog
LD_LIBRARY_PATH=/home/daron/feRcuda/build \
  cargo run --manifest-path rust/apps/fercuda-llm-app/Cargo.toml -- runtime-smoke --alloc-mb 128
LD_LIBRARY_PATH=/home/daron/feRcuda/build \
  cargo run --manifest-path rust/apps/fercuda-llm-app/Cargo.toml -- prepare-session --model qwen2-0.5b-q4
LD_LIBRARY_PATH=/home/daron/feRcuda/build \
  cargo run --manifest-path rust/apps/fercuda-llm-app/Cargo.toml -- \
  mistral-example --cpu --prompt "Explain TLSF allocation in one sentence."
LD_LIBRARY_PATH=/home/daron/feRcuda/build \
  cargo run --manifest-path rust/apps/fercuda-llm-app/Cargo.toml -- \
  mistral-example --device 0 --prompt "Explain TLSF allocation in one sentence."
```

Notes:
- `libptx_core.so` and related native libs must be on `LD_LIBRARY_PATH`
- runtime commands require a machine where the CUDA driver matches the runtime/toolkit used to build `feRcuda`
- the Mistral example now builds against the repo-local PTX/CUDA stack; actual execution still depends on having a local/cached GGUF plus a working CUDA runtime on the target machine
