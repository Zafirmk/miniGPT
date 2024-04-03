from torch.utils.data import Dataset
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from config import get_config
from utils.utils import causal_mask

class LanguageData(Dataset):
    def __init__(self, data_path: str, enc_tokenizer: PreTrainedTokenizerFast, dec_tokenizer: PreTrainedTokenizerFast) -> None:
        super().__init__()
        self.data_path = data_path
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.data = load_dataset("parquet", data_files = self.data_path, split='train')
        self.max_seq_len = get_config()['max_seq_len']

    def __len__(self):
        return self.data.num_rows

    def __getitem__(self, index):

        dec_bos_token = self.dec_tokenizer.bos_token_id
        dec_eos_token = self.dec_tokenizer.eos_token_id
        dec_pad_token = self.dec_tokenizer.pad_token_id

        enc_lang_text = self.data[index]['translation']['en']
        dec_lang_text = self.data[index]['translation']['fr']

        enc_tokenizer_output = self.enc_tokenizer(
            enc_lang_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )

        dec_tokenizer_output = self.dec_tokenizer(
            dec_lang_text,
            max_length=self.max_seq_len-1,
            truncation=True,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_tensors='pt'
        )

        dec_padding_tokens = (self.max_seq_len - len(dec_tokenizer_output['input_ids'].squeeze(0))) - 1

        enc_tokens = enc_tokenizer_output['input_ids'].squeeze(0)
        dec_tokens = torch.cat([
            torch.tensor([dec_bos_token]),
            dec_tokenizer_output['input_ids'].squeeze(0),
        ])
        if dec_padding_tokens > 0:
            dec_tokens = torch.cat([
                dec_tokens,
                torch.tensor([dec_pad_token] * dec_padding_tokens, dtype=torch.int64)
            ])

        enc_mask = enc_tokenizer_output['attention_mask'].unsqueeze(1)
        dec_mask = (dec_tokens != dec_pad_token).int()
        dec_mask = (dec_mask & causal_mask(dec_tokens.size(0)))

        label = torch.cat([
            dec_tokenizer_output['input_ids'].squeeze(0),
            torch.tensor([dec_eos_token], dtype=torch.int64),
            torch.tensor([dec_pad_token] * dec_padding_tokens, dtype=torch.int64)
        ])

        return {
            'enc_lang_text': enc_lang_text,
            'enc_tokens': enc_tokens,
            'enc_mask': enc_mask,

            'dec_lang_text': dec_lang_text,
            'dec_tokens': dec_tokens,
            
            'dec_mask': dec_mask,
            'label': label,   
        }