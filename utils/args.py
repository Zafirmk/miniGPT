import argparse

def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Transformer from scratch", add_help=True)
    parser.add_argument(
        "--d_model",
        default=512,
        type=int,
        help="Dimensionality of embeddings in transformer architecture",
    )
    parser.add_argument(
        "--enc_vocab_size",
        default=30000,
        type=int,
        help="Vocabulary size for encoder vocab",
    )
    parser.add_argument(
        "--dec_vocab_size",
        default=30000,
        type=int,
        help="Vocabulary size for decoder vocab",
    )
    parser.add_argument(
        "--max_seq_len",
        default=350,
        type=int,
        help="Maximum possible length for a character sequence",
    )
    parser.add_argument(
        "--d_hidden",
        default=2048,
        type=int,
        help="Hidden neurons in fully connected layer of transformer architecture",
    )
    parser.add_argument(
        "--num_heads",
        default=8,
        type=int,
        help="Number of attention heads to use in transformer architecture",
    )
    parser.add_argument(
        "--num_blocks",
        default=8,
        type=int,
        help="Number of blocks of encoders and decoders in transformer architecture",
    )
    parser.add_argument(
        "--epochs",
        default=4,
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        default=0.005,
        type=float,
        help="Learning rate for optimizer",
    )

    return parser.parse_args()