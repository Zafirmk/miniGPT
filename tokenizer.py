from datasets import load_dataset
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_all_sentences(data_path: str, language: str):
    data = load_dataset("parquet", data_files=data_path, split='train')
    for item in data:
        yield item[language]

def get_or_build_tokenizer(dpath, lang):
    tokenizer_path = Path(f'./tokenizers/tokenizer-{lang}.json')
    
    # Check if the directory exists, if not create it
    tokenizer_dir = tokenizer_path.parent
    if not tokenizer_dir.exists():
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
    
    if not tokenizer_path.exists():
        # Assuming `get_all_sentences` is a function you've defined to extract sentences from your dataset
        # and `WordLevel`, `Whitespace`, and `WordLevelTrainer` are properly imported
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dpath, lang), trainer=trainer)
        
        # Save the tokenizer
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer