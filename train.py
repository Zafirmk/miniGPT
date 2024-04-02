from config import get_config
from tokenizer import get_or_create_tokenizer
from dataset import LanguageData
from model import create_model, EncoderDecoderTransformer
from torch.utils.data import DataLoader, random_split
import torch

def train():
    config = get_config()

    enc_tokenizer = get_or_create_tokenizer(
        "en", "./tokenizers/enc.json", bos_token='<s>', eos_token='</s>', pad_token='[PAD]', vocab_size=config['enc_vocab_size'], data_path='./data.parquet', batch_size=32
    )

    dec_tokenizer = get_or_create_tokenizer(
        "fr", "./tokenizers/dec.json", bos_token='<s>', eos_token='</s>', pad_token='[PAD]', vocab_size=config['dec_vocab_size'], data_path='./data.parquet', batch_size=32
    )

    data = LanguageData(
        "./data.parquet",
        enc_tokenizer,
        dec_tokenizer,
    )

    total_size = len(data)
    train_size = int(total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        drop_last=True
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=32,
        drop_last=True
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

            print(f"Batch Number: {idx+1}")

            enc_tokens = batch['enc_tokens']
            dec_tokens = batch['dec_tokens']

            enc_mask = batch['enc_mask']
            dec_mask = batch['dec_mask']

            label = batch['label']
            label = label.view(-1)

            pred = model(enc_tokens, dec_tokens, enc_mask, dec_mask)
            pred = pred.view(-1, config['dec_vocab_size'])
            loss = loss_fn(pred, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Run validation after every epoch

        print(f"Epoch {epoch+1} \n Loss: {loss} \n")


if __name__ == "__main__":
    train()