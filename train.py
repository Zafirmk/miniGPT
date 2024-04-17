import os
import numpy as np
import torch.distributed as dist
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
from torch.nn.parallel import DistributedDataParallel
from datasets import load_dataset
import socket

def find_free_port():
    print(4)
    s = socket.socket()
    print(5)
    s.bind(('', 0))  # Bind to a free port provided by the host.
    print(6)
    port = s.getsockname()[1]  # Get the port number
    print(7)
    s.close()
    print(8)
    print(f"PORT: {port}")
    return port

def setup(rank, world_size):
    print(3)
    port = find_free_port()
    print(9)
    init_method = f'tcp://{os.environ.get("MASTER_ADDR")}:{port}'
    print(10)
    print(init_method)
    dist.init_process_group(backend='nccl', init_method=init_method, world_size=world_size, rank=rank)
    print(11)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    print(12)
    
    config = get_config()
    if int(os.environ.get('SLURM_PROCID')) == 0:
        init_stats()

    enc_tokenizer = get_or_build_tokenizer('/home/zafirmk/scratch/miniGPT/data.parquet', 32, 'en')
    dec_tokenizer = get_or_build_tokenizer('/home/zafirmk/scratch/miniGPT/data.parquet', 32, 'fr')

    data = LanguageData(
        "/home/zafirmk/scratch/miniGPT/data.parquet",
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
        num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)),
        sampler=DistributedSampler(train_dataset, num_replicas=world_size, rank=rank),
        shuffle=False
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
    ).to(rank)

    model.train()
    model = DistributedDataParallel(model, device_ids=[rank])

    loss_fn = torch.nn.CrossEntropyLoss().to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    for epoch in range(config['epochs']):
        for idx, batch in enumerate(train_dataloader):

            enc_tokens = batch['enc_tokens'].to(rank)
            dec_tokens = batch['dec_tokens'].to(rank)

            enc_mask = batch['enc_mask'].to(rank)
            dec_mask = batch['dec_mask'].to(rank)

            label = batch['label'].to(rank)

            pred = model(enc_tokens, dec_tokens, enc_mask, dec_mask)
            loss = loss_fn(pred.view(-1, config['dec_vocab_size']), label.view(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f'Epoch {epoch+1} Completed on proc: {os.environ.get("SLURM_PROCID")}')
        
        if int(os.environ.get('SLURM_PROCID')) == 0:
            collect_stats(loss, 0, epoch)