#!/bin/bash
#SBATCH --job-name=multi_gpu
#SBATCH --account=def-eugenium
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G      
#SBATCH --time=01:00:00
#SBATCH --output=logs/single_gpu_%j.out

module purge
module load cuda
module load python/3.11
module load arrow/15.0.1

source ~/projects/def-eugenium/zafirmk/miniGPT/testenv/bin/activate

echo "Starting Training..."

export TORCH_NCCL_BLOCKING_WAIT=1
export MASTER_ADDR=$(hostname)
echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

srun ~/projects/def-eugenium/zafirmk/miniGPT/testenv/bin/python -u main.py