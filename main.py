import os
import torch
import torch.distributed as dist
from train import train

def main():
    ngpus_per_node = torch.cuda.device_count()
    n_tasks = int(os.environ.get("SLURM_NTASKS"))
    local_rank = int(os.environ.get("SLURM_LOCALID")) # Local Process ID
    global_rank = int(os.environ.get("SLURM_PROCID")) # Global Process ID
    job_id = int(os.environ.get("SLURM_JOBID"))
    n_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES"))
    node_id = int(os.environ.get("SLURM_NODEID"))
    available_gpus = list(os.environ.get('CUDA_VISIBLE_DEVICES').replace(',',""))
    host = os.environ.get('HOSTNAME')

    print(f'ngpus_per_node: {ngpus_per_node}')
    print(f'n_tasks: {n_tasks}')
    print(f'local_id: {local_rank}')
    print(f'proc_id: {global_rank}')
    print(f'job_id: {job_id}')
    print(f'n_nodes: {n_nodes}')
    print(f'node_id: {node_id}')
    print(f'available_gpus: {available_gpus}')
    print(f'host: {host}')
    print('\n')

if __name__ == "__main__":
    main()