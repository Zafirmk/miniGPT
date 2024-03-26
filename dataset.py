from torch.utils.data import Dataset
import torch
from datasets import load_dataset

class EnglishFrenchData(Dataset):
    def __init__(self, data_path, enc_tokenizer, dec_tokenizer) -> None:
        super().__init__()
        self.data_path = data_path
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.data = load_dataset("parquet", data_files = self.data_path, split='train')

    def __len__(self):
        return self.data['train'].num_rows

    def __getitem__(self, index):
        enc_lang_text = self.data[index]['translation']['en']
        dec_lang_text = self.data[index]['translation']['fr']

        enc_tokens = torch.tensor([0]) # enc_tokenizer will encode here
        dec_tokens = torch.tensor([0]) # dec_tokenizer will encode here

        enc_mask = torch.tensor([-1])
        dec_mask = torch.tensor([-1])

        label = torch.tensor([1])

        return {
            'enc_tokens': enc_tokens,
            'dec_tokens': dec_tokens,
            'enc_mask': enc_mask,
            'dec_mask': dec_mask,
            'label': label
        }

data = EnglishFrenchData("./data.parquet", "", "")
print(data[0])