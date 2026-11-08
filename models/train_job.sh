#!/bin/bash
# Example Slurm job for train.py. Submit from the models/ folder with:  sbatch train_job.sh
# Lines marked EDIT depend on your cluster (check its docs for partition and module names).

#SBATCH --job-name=cxr-vinbig
#SBATCH --partition=gpu            # EDIT: your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out    # create logs/ before submitting: mkdir -p logs

set -euo pipefail

# EDIT: load your Python/conda setup, then activate the environment that has
# tensorflow[and-cuda], pandas, scikit-learn, scipy, opencv-python-headless and pillow
module load python                 # EDIT
source activate cxr                # EDIT (or: conda activate cxr / source venv/bin/activate)

DATA_DIR=$HOME/GeneralizedCXR/data # EDIT: where you uploaded the data folder
SEED=${SEED:-0}                    # override per job: sbatch --export=ALL,SEED=1 train_job.sh

nvidia-smi
python train.py \
    --data-dir "$DATA_DIR" \
    --out-dir "runs/vinbig_seed${SEED}" \
    --seed "$SEED" \
    --batch-size 32
