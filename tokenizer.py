from tokenizers import decoders, models, pre_tokenizers, processors, trainers, Tokenizer
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from pathlib import Path
import torch
class CustomTokenizer():
    def __init__(self, language: str, vocab_size: int, data_path: str, batch_size: int, tokenizer_save_path: str, bos_token: str, eos_token: str, pad_token: str) -> None:
        self.language = language
        self.data_path = data_path
        self.batch_size = batch_size
        self.special_tokens = [token for token in [bos_token, eos_token, pad_token] if token is not None]

        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=self.special_tokens)
        self.train()
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{bos_token}:0 $A:0" + (f" {eos_token}:0" if eos_token is not None else ""),
            special_tokens=list(zip(self.special_tokens, list(map(lambda x: self.tokenizer.token_to_id(x), self.special_tokens))))
        )
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.save(tokenizer_save_path)

    def batch_iterator(self, data_path: str, batch_size: int):
        data = load_dataset("parquet", data_files=data_path, split='train')
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size][self.language]

    def train(self):
        self.tokenizer.train_from_iterator(
            self.batch_iterator(
                self.data_path,
                self.batch_size
            ),
            trainer=self.trainer
        )


def get_or_create_tokenizer(language: str, tokenizer_path: str, bos_token: str = None, eos_token: str = None, pad_token: str = None, vocab_size: int = None, data_path: str = None, batch_size: int = None) -> PreTrainedTokenizerFast:
    file_path = Path(tokenizer_path)
    
    if not file_path.exists():
        CustomTokenizer(
            language=language,
            vocab_size=vocab_size,
            data_path=data_path,
            batch_size=batch_size,
            tokenizer_save_path=tokenizer_path,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token
        )
    
    return (
            PreTrainedTokenizerFast(
                tokenizer_file=tokenizer_path,
                bos_token=bos_token,
                eos_token=eos_token,
                pad_token=pad_token
            )
        )