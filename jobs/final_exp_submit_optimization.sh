#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
sbatch jobs/final_exp_V2_M1_optim_seed43.sh
sbatch jobs/final_exp_V2_M1_optim_seed44.sh
sbatch jobs/final_exp_V2_M1_optim_seed45.sh
