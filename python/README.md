# feRcuda mega test wheel

Pre-built wheel for the feRcuda B200 TLSF stress test. Build on a machine with cmake, nvcc, rust; install on clusters with only Python.

## Build (machine with module load cmake cuda gcc)

```bash
# Load build tools (adjust for your cluster)
module load cmake cuda gcc  # or: module load cmake cuda/13.1 gcc

# Build and create wheel
./scripts/build_wheel.sh
```

Output: `python/dist/feRcuda_mega_test-0.1.0-*.whl`

## Install (cluster with only Python)

```bash
pip install feRcuda_mega_test-*.whl
pip install torch  # if not already installed
```

## Run

```bash
feRcuda-mega-test
# or
python -m feRcuda_mega_test

# richer demos
feRcuda-mega-test-long
python -m feRcuda_mega_test long

feRcuda-mega-test-extreme
python -m feRcuda_mega_test extreme
```
