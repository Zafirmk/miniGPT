import torch
from torch.distributed import init_process_group, destroy_process_group
from train import Trainer
from utils.model_utils import create_training_objs

def main():
    init_process_group('nccl')
    model, train_dataloader, _, loss_fn, optimizer = create_training_objs()
    trainer = Trainer(model, optimizer, loss_fn, train_dataloader)
    trainer.train()
    destroy_process_group()

if __name__ == "__main__":
    main()