import wandb
import os
import torch
import numpy as np
import torch.utils
from config import get_config
from tokenizer import get_or_build_tokenizer
from dataset import LanguageData
from model import create_model, EncoderDecoderTransformer
from torch.utils.data import DataLoader, random_split
import torch
from torch.utils.data.distributed import DistributedSampler

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

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
        batch_size=96,
        drop_last=True,
        shuffle=False,
        sampler=DistributedSampler(train_dataset),
        num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=128,
        drop_last=True,
        shuffle=False,
        sampler=DistributedSampler(val_dataset),
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

    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    return model, train_dataloader, val_dataloader, loss_fn, optimizer

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, epoch, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["enc_tokens"].to(device) # (b, seq_len)
            encoder_mask = batch["dec_tokens"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["enc_lang_text"]
            target_text = batch["dec_lang_text"]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")
    
            # Evaluate the character error rate
            # Compute the char error rate 
            metric = torchmetrics.CharErrorRate()
            cer = metric(predicted, expected)
            wandb.log(
            {
                f"Character error rate": cer,
            },
                step=epoch
            )


            # Compute the word error rate
            metric = torchmetrics.WordErrorRate()
            wer = metric(predicted, expected)
            wandb.log(
            {
                f"Character error rate": wer,
            },
                step=epoch
            )

            # Compute the BLEU metric
            metric = torchmetrics.BLEUScore()
            bleu = metric(predicted, expected)
            wandb.log(
            {
                f"Character error rate": bleu,
            },
                step=epoch
            )

            if count == num_examples:
                break