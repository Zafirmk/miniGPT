import os
import numpy as np
import torch.distributed
from config import get_config
from tokenizer import get_or_build_tokenizer
from dataset import LanguageData
from model import create_model, EncoderDecoderTransformer
from utils.log import collect_stats, init_stats
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch
from utils.model_utils import estimate_total_gpu_usage
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset

def train():
    config = get_config()
    init_stats()

    enc_tokenizer = get_or_build_tokenizer('./data.parquet', 32, 'en')
    dec_tokenizer = get_or_build_tokenizer('./data.parquet', 32, 'fr')

    data = LanguageData(
        "./data.parquet",
        enc_tokenizer,
        dec_tokenizer,
    )

    total_size = len(data)
    train_size = int(total_size * 0.99)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        drop_last=True,
        num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    )

    model = create_model(
        EncoderDecoderTransformer,
        d_model = config['d_model'],
        enc_vocab_size = config['enc_vocab_size'],
        dec_vocab_size = config['dec_vocab_size'],
        max_seq_len = config['max_seq_len'],
        d_hidden = config['d_hidden'],
        num_heads = config['num_heads'],
        num_blocks = config['num_blocks']
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.train()

    for epoch in range(config['epochs']):
        for idx, batch in enumerate(train_dataloader):

            enc_tokens = batch['enc_tokens']
            dec_tokens = batch['dec_tokens']

            enc_mask = batch['enc_mask']
            dec_mask = batch['dec_mask']

            label = batch['label']

            pred = model(enc_tokens, dec_tokens, enc_mask, dec_mask)
            loss = loss_fn(pred.view(-1, config['dec_vocab_size']), label.view(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        collect_stats(loss, 0, epoch)