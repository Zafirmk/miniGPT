from utils.args import get_args_parser

def get_config() -> dict:
    args = get_args_parser()

    config = {
        "d_model": args.d_model, # 512
        "enc_vocab_size": args.enc_vocab_size, #30,000
        "dec_vocab_size": args.dec_vocab_size, #30,000
        "max_seq_len": args.max_seq_len, #350
        "d_hidden": args.d_hidden, #2048
        "num_heads": args.num_heads, #8
        "num_blocks": args.num_blocks, #8
        "epochs": args.epochs #4
    }
    return config