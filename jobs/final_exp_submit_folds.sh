#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
sbatch jobs/final_exp_V2_M1_fold1.sh
sbatch jobs/final_exp_V2_M1_fold2.sh
sbatch jobs/final_exp_V2_M1_fold3.sh
