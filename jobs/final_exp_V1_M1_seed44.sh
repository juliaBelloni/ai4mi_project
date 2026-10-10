#!/bin/bash
#SBATCH --job-name=final_exp_V1_M1_seed44_prep
#SBATCH --partition=rome
#SBATCH --cpus-per-task=10
#SBATCH --time=01:00:00
#SBATCH --account=gisr128855
#SBATCH --output=logs/slurm_%x_%j.out

set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/ai4mi_project}}"
NAME="final_exp_V1_M1_seed44"
SCRIPT="$REPO/jobs/$NAME.sh"
SOURCE_DIR="$REPO/data/segthor_train_full"
DATA_DIR="$REPO/data/segthor_full_seed43/final_exp_A2_A1_crop"
RESULT_DIR="$REPO/results/segthor_full_seed43/$NAME"
LOG_DIR="$REPO/logs"
REFERENCE="$REPO/results/segthor_full_seed43/final_exp_M1_A2_unet_large"
CONFIG="$RESULT_DIR/postprocessing/dense_crf/postprocessing_config.json"
PP=(--run_dir "$RESULT_DIR" --data_dir "$DATA_DIR" --source_dir "$SOURCE_DIR" --reference_run "$REFERENCE")

module load 2023 Python/3.11.3-GCCcore-12.3.0
cd "$REPO"
source ai4mi/bin/activate
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=44 CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLBACKEND=Agg

case "${1:-prepare}" in
    prepare)
        mkdir -p "$LOG_DIR"
        [[ -f "$DATA_DIR/preprocessing.json" && -d "$DATA_DIR/val/geometry" ]] || { echo "Missing completed A2 data/geometry" >&2; exit 1; }
        for method in dense_crf anatomy_safe_fill crf_aorta_anatomy12_fill; do
            python jobs/final_exp_postprocess.py "${PP[@]}" --method "$method" --prepare-only
        done
        sbatch --job-name="${NAME}_train" --partition=gpu_a100 --gpus=1 \
            --cpus-per-task=16 --time=05:00:00 \
            --output="$LOG_DIR/slurm_%x_%j.out" "$SCRIPT" train
        ;;
    train)
        python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
        python main.py --dataset SEGTHOR --mode full --model unet-large --epochs 25 \
            --data_dir "$DATA_DIR" --dest "$RESULT_DIR/model" --gpu \
            --augment --augment_scale 0.15 --context_slices 2 --loss_fn balance \
            --balance_inter binary --balance_normalized none --balance_alpha 0.5 \
            --balance_t 0.9 --balance_fallback_epoch 12 --opt adam --lr 0.0005 \
            --scheduler none --deterministic --seed 44
        sbatch --job-name="${NAME}_export" --partition=gpu_a100 --gpus=1 \
            --cpus-per-task=10 --time=01:00:00 --mem=32G \
            --output="$LOG_DIR/slurm_%x_%j.out" "$SCRIPT" export
        ;;
    export)
        python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
        python export_probabilities.py --result_dir "$RESULT_DIR/model" --data_dir "$DATA_DIR" \
            --geometry_dir "$DATA_DIR/val/geometry" --postprocessing_config "$CONFIG" --gpu
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
            --geometry_dir "$DATA_DIR/val/geometry" \
            --grp_regex '(Patient_[0-9]+)_[0-9]+' \
            --source_scan_pattern "$SOURCE_DIR/train/{id_}/GT.nii.gz"
        python eval.py --pred_folder "$RESULT_DIR/pred_volumes" \
            --gt_pattern "$SOURCE_DIR/train/{id_}/GT.nii.gz" --num_classes 5 \
            --metrics dice hausdorff_distance_95 average_surface_distance \
            --confusion_matrix --dest "$RESULT_DIR/eval_metrics.csv"
        sbatch --job-name="${NAME}_post" --partition=rome --cpus-per-task=10 \
            --time=06:00:00 --mem=64G --output="$LOG_DIR/slurm_%x_%j.out" "$SCRIPT" postprocess
        ;;
    postprocess)
        export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
        for method in anatomy_safe_fill dense_crf crf_aorta_anatomy12_fill; do
            python jobs/final_exp_postprocess.py "${PP[@]}" --method "$method"
        done
        ;;
    *) echo "Unknown stage: $1" >&2; exit 2 ;;
esac
