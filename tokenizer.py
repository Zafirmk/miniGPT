from tokenizers import decoders, models, pre_tokenizers, processors, trainers, Tokenizer
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from pathlib import Path
class CustomTokenizer():
    def __init__(self, language: str, vocab_size: int, data_path: str, batch_size: int, special_tokens: list[str], tokenizer_save_path: str) -> None:
        self.language = language
        self.data_path = data_path
        self.batch_size = batch_size

        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        self.train()
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{special_tokens[0]}:0 $A:0 {special_tokens[1]}:0",
            special_tokens=list(zip(special_tokens, list(map(lambda x: self.tokenizer.token_to_id(x), special_tokens))))
        )
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.save(tokenizer_save_path)

    def batch_iterator(self, data_path: str, batch_size: int):
        data = load_dataset("parquet", data_files=data_path, split='train')
        for i in range(0, len(data), batch_size):
            yield (list(map(lambda x: x[self.language], data[i : i + batch_size]['translation'])))

    def train(self):
        self.tokenizer.train_from_iterator(
            self.batch_iterator(
                self.data_path,
                self.batch_size
            ),
            trainer=self.trainer
        )


def get_or_create_tokenizer(language: str, tokenizer_path: str, vocab_size: int = None, data_path: str = None, batch_size: int = None, special_tokens: list[str] = []) -> PreTrainedTokenizerFast:
    file_path = Path(tokenizer_path)
    
    if not file_path.exists():
        CustomTokenizer(
            language=language,
            vocab_size=vocab_size,
            data_path=data_path,
            batch_size=batch_size,
            special_tokens=special_tokens,
            tokenizer_save_path=tokenizer_path
        )
    
    return (
            PreTrainedTokenizerFast(
                tokenizer_file=tokenizer_path,
                pad_token=special_tokens[-1],
            )
        )

enc_tokenizer = get_or_create_tokenizer(
    "en", "./tokenizers/enc.json", 25000, "./data.parquet", 32, ['<s>', '</s>', '[PAD]']
)

dec_tokenizer = get_or_create_tokenizer(
    "fr", "./tokenizers/dec.json", 25000, "./data.parquet", 32, ['<s>', '</s>', '[PAD]']
)

print(enc_tokenizer(
    'Hello my name is zafir',
    max_length=128,
    truncation=True,
    padding='max_length',
    add_special_tokens=True,
    return_token_type_ids=False,
    return_tensors='pt'
)['input_ids'])

print("\n--------------------------------------------------------\n")

print(dec_tokenizer(
    'Hello my name is zafir',
    max_length=128,
    truncation=True,
    padding='max_length',
    add_special_tokens=False,
    return_token_type_ids=False,
    return_tensors='pt'
)['input_ids'])
