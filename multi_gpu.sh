#!/bin/bash
#SBATCH --job-name=multi_gpu
#SBATCH --account=def-eugenium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/multi_gpu_%j.out

module purge
module load cuda
module load python/3.11
module load arrow/15.0.1

source ~/projects/def-eugenium/zafirmk/miniGPT/testenv/bin/activate

echo "Starting Training..."

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Head Node IP: $head_node_ip
echo Nodes Array: $nodes_array
export LOGLEVEL=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

srun ~/projects/def-eugenium/zafirmk/miniGPT/testenv/bin/torchrun \
 --standalone \
 --nproc_per_node=gpu \
 main.py