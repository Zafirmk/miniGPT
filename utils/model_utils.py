import os
from tqdm import tqdm
import torch
import numpy as np
import torch.utils
from config import get_config
from tokenizer import get_or_build_tokenizer
from dataset import LanguageData, causal_mask
from model import create_model, EncoderDecoderTransformer
from torch.utils.data import DataLoader, random_split
import torch
from torchmetrics.text import CharErrorRate
from torch.utils.data.distributed import DistributedSampler

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

    enc_tokenizer = get_or_build_tokenizer('/home/zafirmk/scratch/miniGPT/data.parquet', 'en')
    dec_tokenizer = get_or_build_tokenizer('/home/zafirmk/scratch/miniGPT/data.parquet', 'fr')

    data = LanguageData(
        "/home/zafirmk/scratch/miniGPT/data.parquet",
        enc_tokenizer,
        dec_tokenizer,
    )

    total_size = len(data)
    train_size = int(total_size * 0.99)
    val_size = total_size - train_size

    print(f"[GPU:{int(os.environ['RANK'])}] | Train dataset size: {train_size}")
    print(f"[GPU:{int(os.environ['RANK'])}] | Val dataset size: {val_size}")

    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        drop_last=False,
        shuffle=False,
        sampler=DistributedSampler(train_dataset),
        num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        drop_last=False,
        shuffle=False,
        sampler= DistributedSampler(val_dataset),
        num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    )

    model = create_model(
        EncoderDecoderTransformer,
        d_model = config['d_model'],
        enc_vocab_size = enc_tokenizer.get_vocab_size(),
        dec_vocab_size = dec_tokenizer.get_vocab_size(),
        max_seq_len = config['max_seq_len'],
        d_hidden = config['d_hidden'],
        num_heads = config['num_heads'],
        num_blocks = config['num_blocks']
    )

    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=enc_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    return model, train_dataloader, val_dataloader, loss_fn, optimizer, dec_tokenizer

def validate_batch(model, batch, device):

    dec_tokenizer = get_or_build_tokenizer('/home/zafirmk/scratch/miniGPT/data.parquet', 'fr')
    sos_idx = dec_tokenizer.token_to_id('[SOS]')
    eos_idx = dec_tokenizer.token_to_id('[EOS]')

    enc_tokens = batch['enc_tokens'].to(device)
    enc_mask = batch['enc_padding_mask'].to(device)

    max_len = get_config()['max_seq_len']

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.module.encode(enc_tokens, enc_mask).to(device)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(enc_tokens).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(enc_mask).to(device)

        # calculate output
        out = model.module.decode(decoder_input, encoder_output, encoder_output, enc_mask, decoder_mask).to(device)

        # get next token
        prob = model.module.project(out[:, -1]).to(device)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(enc_tokens).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def validation(model: EncoderDecoderTransformer, val_dataloader: DataLoader, num_examples=1):
    model.eval()
    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    dec_tokenizer = get_or_build_tokenizer('/home/zafirmk/scratch/miniGPT/data.parquet', 'fr')
    cer = CharErrorRate()
    total = 0
    count = 0

    for batch in val_dataloader:
        x = validate_batch(model, batch, int(os.environ['LOCAL_RANK'])).detach().cpu().numpy()

        pred_sentence = dec_tokenizer.decode(x)

        total += cer(pred_sentence, batch["dec_lang_text"][0])
        count += 1

        if count < num_examples:
            print('-'*console_width)
            print(f"[GPU:{int(os.environ['RANK'])}] | {f'SOURCE: ':>12}{batch['enc_lang_text'][0]}")
            print(f"[GPU:{int(os.environ['RANK'])}] | {f'TARGET: ':>12}{batch['dec_lang_text'][0]}")
            print(f"[GPU:{int(os.environ['RANK'])}] | {f'PREDICTED: ':>12}{pred_sentence}")
            print('-'*console_width)

        if count % 10 == 0:
            print(f"[GPU:{int(os.environ['RANK'])}] | Processed {count+1} validation examples")

    return (total / count)