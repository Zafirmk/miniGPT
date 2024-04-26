import os
import wandb
from utils.args import get_args_parser

def collect_stats(train_loss, val_loss, epoch):
    wandb.log(
        {
            f"Training Loss": train_loss,
            f"Validation Loss": val_loss,
        },
        step=epoch
    )

def init_stats():
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    WANDB_USERNAME = os.environ.get("WANDB_USERNAME")
    
    wandb.init(
            project="miniGPT",
            mode="online",
            config=get_args_parser()
    )