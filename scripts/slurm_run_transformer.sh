#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -J fercuda-transformer
#SBATCH --output=fercuda-transformer-%j.out

export HOME="${HOME:-$(eval echo ~$(whoami))}"
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
"$HOME/.local/bin/feRcuda-transformer"
