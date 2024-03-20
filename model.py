import torch
import torch.nn as nn

class WordEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding_table = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model
        )
    
    def forward(self, x):
        return self.embedding_table(x) * torch.sqrt(torch.tensor(self.d_model))
