from config import get_config
from tokenizer import get_or_create_tokenizer
from dataset import LanguageData
from model import create_model, EncoderDecoderTransformer
from torch.utils.data import DataLoader

if __name__ == "__main__":
    config = get_config()

    enc_tokenizer = get_or_create_tokenizer(
        "en", "./tokenizers/enc.json", bos_token='<s>', eos_token='</s>', pad_token='[PAD]', vocab_size=config['vocab_size'], data_path='./data.parquet', batch_size=32
    )

    dec_tokenizer = get_or_create_tokenizer(
        "fr", "./tokenizers/dec.json", bos_token='<s>', eos_token='</s>', pad_token='[PAD]', vocab_size=config['vocab_size'], data_path='./data.parquet', batch_size=32
    )

    data = LanguageData(
        "./data.parquet",
        enc_tokenizer,
        dec_tokenizer
    )

    train_dataloader = DataLoader(
        dataset=data,
        batch_size=16,
        drop_last=True
    )

    sample = (next(iter(train_dataloader)))

    enc_tokens = sample['enc_tokens']
    dec_tokens = sample['dec_tokens']
    enc_mask = sample['enc_mask']
    dec_mask = sample['dec_mask']
    dec_text = sample['dec_lang_text']

    model = create_model(
        EncoderDecoderTransformer,
        d_model = config['d_model'],
        vocab_size = config['vocab_size'],
        max_seq_len = config['max_seq_len'],
        d_hidden = config['d_hidden'],
        num_heads = config['num_heads'],
        num_blocks = config['num_blocks']
    )

    x = model(enc_tokens, dec_tokens, enc_mask, dec_mask)

    print(x)
    total = sum(p.numel() for p in model.parameters())
    print(total)
