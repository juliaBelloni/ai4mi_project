#!/bin/bash
#SBATCH --job-name=final_exp_Q11_small_components
#SBATCH --partition=rome
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=gisr128855
#SBATCH --output=logs/slurm_%x_%j.out

set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/ai4mi_project}}"
module load 2023 Python/3.11.3-GCCcore-12.3.0
cd "$REPO"
source ai4mi/bin/activate
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10

python jobs/final_exp_postprocess.py --method small_components
