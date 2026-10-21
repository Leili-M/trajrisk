#!/bin/bash

#SBATCH --job-name=train

#SBATCH --account=project_2021634

#SBATCH --partition=gpumedium

#SBATCH --time=01:00:00

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=72

#SBATCH --gres=gpu:gh200:1

#SBATCH --output=/scratch/project_2021634/logs/%x_%j.out



# Usage: sbatch jobs/train.sh <config file> <run name> [more options for train_diffrec.py]

# Example: sbatch jobs/train.sh configs/movielens.yaml ml_base

set -eo pipefail



module load python-pytorch/2.13

cd /projappl/project_2021634/trajrisk

export PYTHONPATH=src
export PYTHONUNBUFFERED=1

export TRAJRISK_STORAGE_ROOT=/scratch/project_2021634

export TRAJRISK_CONFIG="$1"

RUN="$2"

shift 2



echo "Job $SLURM_JOB_ID on $(hostname) | code $(git rev-parse --short HEAD) | config $TRAJRISK_CONFIG | run $RUN"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader



srun python scripts/train_diffrec.py --run-name "$RUN" "$@"

