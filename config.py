config = {
    "d_model": 512,
    "enc_vocab_size": 141,
    "dec_vocab_size": 150,
    "max_seq_len": 350,
    "d_hidden": 2048,
    "num_heads": 8,
    "num_blocks": 2,
    "epochs": 20
}

def get_config() -> dict:
    return config