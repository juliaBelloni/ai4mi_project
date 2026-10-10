#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
sbatch jobs/final_exp_V1_M1_seed44.sh
sbatch jobs/final_exp_V1_M1_seed45.sh
