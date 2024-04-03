import os

import torch.distributed
from config import get_config
from tokenizer import get_or_create_tokenizer
from dataset import LanguageData
from model import create_model, EncoderDecoderTransformer
from transformers import PreTrainedTokenizerFast
from utils.utils import causal_mask, get_or_build_tokenizer, collect_stats, init_stats
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset

def train():
    if global_rank == 0:
        init_stats()

    config = get_config()
    ds_raw = load_dataset("parquet", data_files='./data.parquet', split='train')

    enc_tokenizer = get_or_build_tokenizer(config, ds_raw, 'en')
    dec_tokenizer = get_or_build_tokenizer(config, ds_raw, 'fr')

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
        sampler=DistributedSampler(
            train_dataset,
            shuffle=True
        )
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

    if os.path.exists('latest_checkpoint.pth'):
        model.load_state_dict(torch.load('latest_checkpoint.pth'))

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
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
        
        if global_rank == 0:
            collect_stats()
        
        # Run validation after every epoch
        # validation(model, val_dataloader, dec_tokenizer)
        print(f"Epoch {epoch+1} \n Loss: {loss} \n")

if __name__ == "__main__":
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['GLOBAL_RANK'])

    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    train()

    torch.distributed.destroy_process_group()