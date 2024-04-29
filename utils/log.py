import os
import wandb
from utils.args import get_args_parser

def collect_stats(train_loss, cer_loss, wer_loss, bleu_loss, epoch):
    wandb.log(
        {
            f"train/Training Loss": train_loss,
            f"validation/CER Loss": cer_loss,
            f"validation/WER Loss": wer_loss,
            f"validation/BLEU Loss": bleu_loss,
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