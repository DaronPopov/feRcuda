#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -J fercuda-extreme
#SBATCH --output=fercuda-extreme-%j.out

export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
feRcuda-mega-test-extreme
