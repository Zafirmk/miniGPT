import os
import torch
import torch.distributed as dist
from train import train

def main():
    gpu_id = torch.cuda.current_device()
    proc_id = os.environ.get('SLURM_PROCID')
    host = os.environ.get('HOSTNAME')

    print(f'\nStarting process: {proc_id} on GPU: {gpu_id}\n')

    # Single GPU, Multiple process
    dist.init_process_group(backend="mpi", init_method=f'tcp://{host}:8921')

    train()


if __name__ == "__main__":
    main()