import torch
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import load_dataset
from config import get_config

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def collect_stats():
    pass

def init_stats():
    pass

def greedy_decode(model, batch: dict, config: dict, dec_tokenizer):
    # Encode the input
    enc_out = model.encode(batch['enc_tokens'], batch['enc_mask'])

    bos_token_id = dec_tokenizer.token_to_id('[SOS]')
    eos_token_id = dec_tokenizer.token_to_id('[PAD]')

    dec_input = torch.tensor([bos_token_id]).unsqueeze(0)

    while dec_input.size(1) != config['max_seq_len']:

        dec_mask = causal_mask(dec_input.size(1)).int()
        dec_out = model.decode(dec_input, enc_out, enc_out, batch['enc_mask'], dec_mask)
        prob = torch.softmax(model.project(dec_out[:, -1]), dim=1)
        _, next_word = torch.max(prob, dim=1)
        dec_input = torch.cat((
            dec_input,
            next_word.unsqueeze(0)
        ), dim=1)

        if next_word == eos_token_id:
            break
    
    return dec_input.squeeze(0)

def get_all_sentences(data_path: str, batch_size: int, language: str):
    data = load_dataset("parquet", data_files=data_path, split='train')
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size][language]

def get_or_build_tokenizer(config, ds, lang):
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
        tokenizer.train_from_iterator(get_all_sentences('./data.parquet', 32, lang), trainer=trainer)
        
        # Save the tokenizer
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def validation(model, val_dataloader, dec_tokenizer: Tokenizer):
    config = get_config()
    model.eval()

    expected = []
    preds = []

    with torch.no_grad():
        for idx, batch in enumerate(val_dataloader):

            pred = greedy_decode(model, batch, config, dec_tokenizer)
            print(pred)

            # preds.append(dec_tokenizer.decode(pred))
            # expected.append(batch['dec_lang_text'])

            if idx == 20:
                break
    
    # Compute char level loss
    # Compute word level loss
