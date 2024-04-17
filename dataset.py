from torch.utils.data import Dataset
import torch
from datasets import load_dataset
from config import get_config
from utils.model_utils import causal_mask
from tokenizers import Tokenizer

class LanguageData(Dataset):
    def __init__(self, data_path: str, enc_tokenizer: Tokenizer, dec_tokenizer: Tokenizer) -> None:
        super().__init__()
        self.data_path = data_path
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.data = load_dataset("parquet", data_files = self.data_path, split='train')
        self.max_seq_len = get_config()['max_seq_len']

    def __len__(self):
        return self.data.num_rows

    def __getitem__(self, index):

        bos_token = self.dec_tokenizer.token_to_id('[SOS]')
        eos_token = self.dec_tokenizer.token_to_id('[EOS]')
        pad_token = self.dec_tokenizer.token_to_id('[PAD]')

        enc_lang_text = self.data[index]['en'][:self.max_seq_len]
        dec_lang_text = self.data[index]['fr'][:self.max_seq_len]

        enc_output = self.enc_tokenizer.encode(enc_lang_text).ids
        dec_output = self.dec_tokenizer.encode(dec_lang_text).ids

        enc_padding_tokens = (self.max_seq_len - len(enc_output)) - 2
        dec_padding_tokens = (self.max_seq_len - len(dec_output)) - 1

        enc_tokens = torch.cat([
            torch.tensor([bos_token]),
            torch.tensor(enc_output),
            torch.tensor([eos_token]),
            torch.tensor([pad_token] * enc_padding_tokens)
        ])

        dec_tokens = torch.cat([
            torch.tensor([bos_token]),
            torch.tensor(dec_output),
            torch.tensor([pad_token] * dec_padding_tokens)
        ])

        label = torch.cat([
            torch.tensor(dec_output),
            torch.tensor([eos_token]),
            torch.tensor([pad_token] * dec_padding_tokens)
        ])

        return {
            'enc_lang_text': enc_lang_text,
            'enc_tokens': enc_tokens,
            'enc_mask': (enc_tokens != pad_token).unsqueeze(0).unsqueeze(0).int(),

            'dec_lang_text': dec_lang_text,
            'dec_tokens': dec_tokens,
            'dec_mask': (dec_tokens != pad_token).unsqueeze(0).int() & causal_mask(dec_tokens.size(0)),

            'label': label,   
        }