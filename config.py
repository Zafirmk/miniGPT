from utils.args import get_args_parser

def get_config() -> dict:
    args = get_args_parser()

    config = {
        "d_model": args.d_model,
        "max_seq_len": args.max_seq_len,
        "d_hidden": args.d_hidden,
        "num_heads": args.num_heads,
        "num_blocks": args.num_blocks,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "model_path": args.model_path,
        "data_path": "/home/zafirmk/scratch/miniGPT/data.parquet"
    }
    return config