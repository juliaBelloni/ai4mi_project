#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
sbatch jobs/final_exp_Q0_none.sh
sbatch jobs/final_exp_Q1_components.sh
sbatch jobs/final_exp_Q2_anatomy.sh
sbatch jobs/final_exp_Q4_fill_holes.sh
sbatch jobs/final_exp_Q5_closing.sh
sbatch jobs/final_exp_Q6_opening.sh
sbatch jobs/final_exp_Q6b_median.sh
sbatch jobs/final_exp_Q7_dense_crf.sh
sbatch jobs/final_exp_Q10_foreground.sh
sbatch jobs/final_exp_Q11_small_components.sh
