#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
# B200/GB200 = 192GB VRAM; request less than that for system RAM
#SBATCH -J fercuda-extreme
#SBATCH --output=fercuda-extreme-%j.out

export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
feRcuda-mega-test-extreme
