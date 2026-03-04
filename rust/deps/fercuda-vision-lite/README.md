# fercuda-vision-lite

Lightweight CV + sensor polling crate for feRcuda.

## Features
- `cv` (default): grayscale frame ops (`threshold`, `canny`, resize, stats).
- `sensor-serial`: CSV sensor polling over serial ports.
- `camera-v4l`: one-frame capture helper for `/dev/video*` devices.

## Notes
- Designed to stay minimal and composable.
- Hardware features are optional so default builds do not require devices.
