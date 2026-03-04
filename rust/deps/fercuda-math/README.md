# fercuda-math

Math facade for feRcuda.

## Design
- Default math layer: `nalgebra` (control, estimation, transforms, fixed-size matrices).
- Optional heavy dense LA: `faer` (enable with feature `faer`).
- Optional tensor interop with Candle: feature `candle`.

## Features
- `faer`: re-exports `faer::Mat` via `fercuda_math::dense::Mat`.
- `candle`: enables `fercuda_math::candle_bridge` for `nalgebra <-> candle_core::Tensor` helpers.

## Usage
```bash
cargo check
cargo check --features faer
cargo check --features candle
cargo check --features "faer candle"
```
