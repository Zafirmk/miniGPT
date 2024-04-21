import sys
import os
import wandb
import torch
import numpy as np
import torch.utils
from config import get_config
from tokenizer import get_or_build_tokenizer
from dataset import LanguageData
from model import create_model, EncoderDecoderTransformer
from torch.utils.data import DataLoader, random_split
import torch
from torchmetrics.text import CharErrorRate

def estimate_total_gpu_usage(model, data_dict, in_bytes=False):
    model_size = 0
    for param in model.parameters():
        model_size += np.prod(param.size()) * param.element_size()
    for buffer in model.buffers():
        model_size += np.prod(buffer.size()) * buffer.element_size()
    
    input_keys = ['enc_tokens', 'dec_tokens', 'enc_mask', 'dec_mask']
    input_size = sum(np.prod(data_dict[key].size()) * data_dict[key].element_size() for key in input_keys if key in data_dict)
    
    total_size = model_size + input_size
    
    if in_bytes:
        return total_size
    else:
        return total_size / (1024 ** 2)

def create_training_objs():
    config = get_config()

    enc_tokenizer = get_or_build_tokenizer('./data_sm.parquet', 32, 'en')
    dec_tokenizer = get_or_build_tokenizer('./data_sm.parquet', 32, 'fr')

    data = LanguageData(
        "./data_sm.parquet",
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
        shuffle=False,
        # sampler=DistributedSampler(train_dataset),
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        drop_last=True,
        shuffle=False,
        # sampler=DistributedSampler(val_dataset),
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

    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    return model, train_dataloader, val_dataloader, loss_fn, optimizer

#TODO: Rewrite validation function
def validation():
    pass