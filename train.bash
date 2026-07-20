#!/bin/bash
#SBATCH --job-name=ellipses
#SBATCH --partition=short
#SBATCH --time=01:55:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ellipses_%j.out

echo "Job started on:"
hostname
date

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dncnn   # ou un env dédié au nouveau projet

cd /home/tbouru/structured_uncertainty_ellipses_cluster/structured_uncertainty_ellipse

echo "Python used:"
which python
echo "Checking PyTorch:"
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"

python main.py --tag mid_gpu

echo "Job finished:"
date