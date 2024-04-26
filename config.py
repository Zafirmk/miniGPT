from utils.args import get_args_parser

def get_config() -> dict:
    args = get_args_parser()

    config = {
        "d_model": args.d_model, # 512
        "max_seq_len": args.max_seq_len, #350
        "d_hidden": args.d_hidden, #2048
        "num_heads": args.num_heads, #8
        "num_blocks": args.num_blocks, #8
        "epochs": args.epochs, #4
        "learning_rate": args.learning_rate, # 0.005
        "batch_size": args.batch_size # 32
    }
    return config