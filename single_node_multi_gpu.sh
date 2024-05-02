#!/bin/bash
#SBATCH --job-name=single_node_multi_gpu
#SBATCH --account=def-eugenium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/single_node_%j.out

module purge
module load cuda
module load python/3.11
module load arrow/15.0.1

source ~/projects/def-eugenium/zafirmk/miniGPT/testenv/bin/activate

echo "Starting Training..."

export LOGLEVEL=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Single Node | Multi GPU
srun ~/projects/def-eugenium/zafirmk/miniGPT/testenv/bin/torchrun \
 --standalone \
 --nproc_per_node=gpu \
 main.py
