#!/bin/bash
#SBATCH --job-name=final_exp_P1_HU_m310_400_prep
#SBATCH --partition=rome
#SBATCH --cpus-per-task=10
#SBATCH --time=01:00:00
#SBATCH --account=gisr128855
#SBATCH --output=logs/slurm_%x_%j.out

set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/ai4mi_project}}"
NAME="final_exp_P1_HU_m310_400"
SCRIPT="$REPO/jobs/$NAME.sh"
SOURCE_DIR="$REPO/data/segthor_train_full"
DATA_DIR="$REPO/data/segthor_full_seed43/$NAME"
RESULT_DIR="$REPO/results/segthor_full_seed43/$NAME"
LOG_DIR="$REPO/logs"

module load 2023 Python/3.11.3-GCCcore-12.3.0
cd "$REPO"
source ai4mi/bin/activate
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=43 CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLBACKEND=Agg

case "${1:-prepare}" in
    prepare)
        mkdir -p "$LOG_DIR"
        if [[ ! -f "$DATA_DIR/preprocessing.json" ]]; then
            python slice_segthor.py --source_dir "$SOURCE_DIR" --dest_dir "$DATA_DIR" \
                --shape 256 256 --retains 10 --seed 43 --fold 0 \
                --hu_min -310 --hu_max 400 --process "${SLURM_CPUS_PER_TASK:-10}"
        fi
        sbatch --job-name="${NAME}_train" --partition=gpu_a100 --gpus=1 \
            --cpus-per-task=18 --time=05:00:00 \
            --output="$LOG_DIR/slurm_%x_%j.out" "$SCRIPT" train
        ;;
    train)
        python main.py --dataset SEGTHOR --mode full --model unet-small --epochs 25 \
            --data_dir "$DATA_DIR" --dest "$RESULT_DIR/model" --gpu \
            --augment --augment_scale 0 --context_slices 2 --loss_fn balance \
            --balance_inter binary --balance_normalized none --balance_alpha 0.5 \
            --balance_t 0.9 --balance_fallback_epoch 12 --opt adam --lr 0.0005 \
            --scheduler none --deterministic --seed 43
        sbatch --job-name="${NAME}_eval" --partition=rome --cpus-per-task=10 \
            --time=01:00:00 --output="$LOG_DIR/slurm_%x_%j.out" "$SCRIPT" evaluate
        ;;
    evaluate)
        for metric in dice_tra dice_val loss_tra loss_val; do
            python plot.py --metric_file "$RESULT_DIR/model/$metric.npy" \
                --dest "$RESULT_DIR/model/$metric.png" --headless
        done
        python stitch.py --data_folder "$RESULT_DIR/model/best_epoch/val" \
            --dest_folder "$RESULT_DIR/pred_volumes" --num_classes 255 \
            --grp_regex '(Patient_[0-9]+)_[0-9]+' \
            --source_scan_pattern "$SOURCE_DIR/train/{id_}/GT.nii.gz"
        python eval.py --pred_folder "$RESULT_DIR/pred_volumes" \
            --gt_pattern "$SOURCE_DIR/train/{id_}/GT.nii.gz" --num_classes 5 \
            --metrics dice hausdorff_distance_95 average_surface_distance \
            --confusion_matrix --dest "$RESULT_DIR/eval_metrics.csv"
        ;;
    *) echo "Unknown stage: $1" >&2; exit 2 ;;
esac
