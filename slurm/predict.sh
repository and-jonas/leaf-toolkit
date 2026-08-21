#!/bin/bash

#SBATCH --job-name=canopy_predict
#SBATCH --partition=gpu
#SBATCH --array=0-1
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --output=%x_%A_%a_%N.log
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU:"
nvidia-smi

cd ~/leaf-toolkit
source .venv/bin/activate

echo "Running: python 04_predict.py ${SLURM_ARRAY_TASK_ID}"
python predict.py $SLURM_ARRAY_TASK_ID
