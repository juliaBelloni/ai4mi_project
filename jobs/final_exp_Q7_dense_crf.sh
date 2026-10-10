#!/bin/bash
#SBATCH --job-name=final_exp_Q7_dense_crf
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

SCRIPT="$REPO/jobs/final_exp_Q7_dense_crf.sh"
RUN="$REPO/results/segthor_full_seed43/final_exp_M1_A2_unet_large"
DATA="$REPO/data/segthor_full_seed43/final_exp_A2_A1_crop"
CONFIG="$RUN/postprocessing/dense_crf/postprocessing_config.json"

case "${1:-prepare}" in
    prepare)
        python jobs/final_exp_postprocess.py --method dense_crf --prepare-only
        [[ -f "$RUN/model/bestweights.pt" && -f "$RUN/model/config.json" ]]
        sbatch --job-name=final_exp_Q7_export --partition=gpu_a100 --gpus=1 \
            --cpus-per-task=10 --time=01:00:00 --mem=32G \
            --output="$REPO/logs/slurm_%x_%j.out" "$SCRIPT" export
        ;;
    export)
        python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
        python export_probabilities.py --result_dir "$RUN/model" --data_dir "$DATA" \
            --geometry_dir "$DATA/val/geometry" --postprocessing_config "$CONFIG" --gpu
        sbatch --job-name=final_exp_Q7_eval --partition=rome --cpus-per-task=10 \
            --time=04:00:00 --mem=64G --output="$REPO/logs/slurm_%x_%j.out" "$SCRIPT" evaluate
        ;;
    evaluate)
        python jobs/final_exp_postprocess.py --method dense_crf
        ;;
    *) echo "Unknown stage: $1" >&2; exit 2 ;;
esac
