# fercuda-ml-lite

Lightweight ML utility crate for feRcuda.

## Included crates
- `safetensors`
- `memmap2`
- `serde`, `serde_json`
- `rand`, `rand_distr`
- `half`

## Helper APIs
- `inspect_safetensors(path)` for metadata checks.
- `to_json` / `from_json` for small config/data payloads.
- `gaussian_noise_f32` for deterministic sampling.
