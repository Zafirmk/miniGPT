import os
import numpy as np
from config import get_config
from model import EncoderDecoderTransformer
from utils.log import collect_stats, init_stats
from torch.utils.data import DataLoader
import torch
from torch.nn.parallel import DistributedDataParallel
from utils.model_utils import validation
from tokenizer import Tokenizer
from tqdm import tqdm

class Trainer:
    def __init__(self,
                 model: EncoderDecoderTransformer,
                 optimizer: torch.optim.Adam,
                 loss_fn: torch.nn.CrossEntropyLoss,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 dec_tokenizer: Tokenizer,
                 snapshot_pth: str = 'snapshots/latest_snapshot.pt',
                 epochs_run: int = 0
        ) -> None:
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.global_rank = int(os.environ['RANK'])
        self.model = model.to(self.local_rank)
        self.dec_tokenizer = dec_tokenizer
        self.optimizer = optimizer
        self.loss_fn = loss_fn.to(self.local_rank)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.snapshot_pth = snapshot_pth
        self.epochs_run = epochs_run
        self.config = get_config()
        if os.path.exists(self.snapshot_pth):
            print(f"\nLoading Snapshot for GPU:{self.global_rank}")
            self.load_snapshot()
        self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank], find_unused_parameters=True)

    def process_batch(self, batch):
        enc_tokens = batch['enc_tokens'].to(self.local_rank)
        dec_tokens = batch['dec_tokens'].to(self.local_rank)

        enc_mask = batch['enc_padding_mask'].to(self.local_rank)
        dec_mask = batch['dec_padding_mask'].to(self.local_rank)

        label = batch['label'].to(self.local_rank)

        pred = self.model(enc_tokens, dec_tokens, enc_mask, dec_mask).to(self.local_rank)
        loss = self.loss_fn(pred.view(-1, self.dec_tokenizer.get_vocab_size()), label.view(-1))

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def process_epoch(self, epoch):
        self.train_dataloader.sampler.set_epoch(epoch)
        b_sz = (next(iter(self.train_dataloader)))['enc_tokens'].shape[0]
        total_loss = 0
        print(f"Start: [GPU:{self.global_rank}] | Epoch: {epoch} | BSize: {b_sz} | Steps: {len(self.train_dataloader)}")
        batch_iter = (tqdm(self.train_dataloader, desc=f"Processing Epoch {epoch:02d}"))
        for idx, batch in enumerate(batch_iter):
            batch_loss = self.process_batch(batch)
            batch_iter.set_postfix({"loss": f"{batch_loss.item():6.3f}"})
            total_loss += batch_loss
        print(f"Complete [GPU:{self.global_rank}] | Epoch: {epoch} | Loss: {total_loss / len(self.train_dataloader)}")
        return total_loss / len(self.train_dataloader)

    def save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch
        }

        torch.save(snapshot, self.snapshot_pth)
        print(f"[GPU:{self.global_rank}] | Epoch {epoch} | Training snapshot saved at {self.snapshot_pth}")

    def load_snapshot(self):
        snapshot = torch.load(self.snapshot_pth, map_location=f"cuda:{self.local_rank}")
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"[GPU:{self.global_rank}] | Resuming training from snapshot at Epoch: {self.epochs_run}")

    def train(self):
        if self.local_rank == 0:
            init_stats()
        for epoch in range(self.epochs_run, self.config["epochs"]):
            train_loss = self.process_epoch(epoch)
            val_loss = validation(self.model, self.val_dataloader)
            if self.local_rank == 0:
                collect_stats(train_loss, val_loss, epoch)
                if epoch % 250 == 0:
                    self.save_snapshot(epoch)
