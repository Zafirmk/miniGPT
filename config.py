config = {
    "d_model": 512,
    "enc_vocab_size": 30000,
    "dec_vocab_size": 30000,
    "max_seq_len": 350,
    "d_hidden": 2048,
    "num_heads": 8,
    "num_blocks": 8,
    "epochs": 4
}

def get_config() -> dict:
    return config