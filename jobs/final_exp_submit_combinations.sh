#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
sbatch jobs/final_exp_Q2a_anatomy_safe.sh
sbatch jobs/final_exp_Q9a_anatomy_fill.sh
sbatch jobs/final_exp_Q7a_crf_aorta.sh
sbatch jobs/final_exp_Q9b_crf_anatomy_fill.sh
