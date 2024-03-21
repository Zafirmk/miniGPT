import torch
import torch.nn as nn
torch.manual_seed(1)

class WordEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding_table = nn.Embedding(
            embedding_dim=self.d_model,
            num_embeddings=self.vocab_size,
        )
    
    def forward(self, x):
        return self.embedding_table(x) * torch.sqrt(torch.tensor(self.d_model))

class PositionalEmbeddings(nn.Module):
    def __init__(self, d_model, seq_len, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.seq_len = seq_len

        p_embed_table = torch.zeros((self.seq_len, self.d_model))
        position = torch.arange(0, seq_len).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(-1 * (torch.arange(0, d_model, 2).float() * (torch.log(torch.tensor(10000.0)) / self.d_model))).unsqueeze(0) # (1, d_model)

        p_embed_table[:, 0::2] = torch.sin(position * div_term)
        p_embed_table[:, 1::2] = torch.cos(position * div_term)

        p_embed_table = p_embed_table.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('p_embed_table', p_embed_table)
    
    def forward(self, x):
        with torch.no_grad():
            return x + self.p_embed_table[:, :x.shape[1], :]
class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.relu = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(d_model, d_hidden)
        self.linear2 = torch.nn.Linear(d_hidden, d_model)
    
    # (batch, seq_len, d_model) --> (batch, seq_len, d_hidden) --> (batch, seq_len, d_model)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class ResidualConnection(nn.Module):
    def __init__(self, d_model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_norm = torch.nn.LayerNorm(d_model)
    
    def forward(self, x, sublayer):
        f = sublayer(x)
        return self.layer_norm(torch.add(f, x))

class MultiHeadAttention(nn.Module):
    pass


# we = WordEmbeddings(4, 24)
# pe = PositionalEmbeddings(4, 10)
# ff = FeedForward(4, 20)
# rs = ResidualConnection(4)

# s = "I wonder what will come next"
# tokens = torch.LongTensor([[11, 23, 21, 22, 5, 15]])

# word_embed = we(tokens)
# pos_embed = pe(word_embed)
# post_rs = rs(pos_embed, ff)


# print(post_rs)