import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(1)

class WordEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, *args, **kwargs) -> None:
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
    def __init__(self, d_model: int, seq_len: int, *args, **kwargs) -> None:
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
    def __init__(self, d_model: int, d_hidden: int, *args, **kwargs) -> None:
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
    def __init__(self, d_model: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_norm = torch.nn.LayerNorm(d_model)
    
    def forward(self, x, sublayer):
        f = sublayer(x)
        return self.layer_norm(torch.add(f, x))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mask: torch.Tensor = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model / num_heads
        self.softmax = torch.nn.Softmax(dim=-1)
        self.d_model = d_model
        self.num_heads = num_heads
        self.mask = mask

        self.w_q = torch.randn((self.d_model, self.d_model))
        self.w_k = torch.randn((self.d_model, self.d_model))
        self.w_v = torch.randn((self.d_model, self.d_model))
        self.w_o = torch.randn((self.d_model, self.d_model))
    
    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        numerator = q @ k.transpose(-2, -1)
        denominator = np.sqrt(self.d_k)
        attention = self.softmax(numerator/denominator)
        if self.mask:
            attention = attention @ self.mask
        return attention @ v

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q_ = q @ self.w_q
        k_ = k @ self.w_k
        v_ = v @ self.w_v

        # (batch, seq, d_model) ---> (batch, num_heads, seq, d_k)
        q_ = q_.view(q_.shape[0], self.num_heads, q.shape[1], int(self.d_k))
        k_ = k_.view(k_.shape[0], self.num_heads, k.shape[1], int(self.d_k))
        v_ = v_.view(v_.shape[0], self.num_heads, v.shape[1], int(self.d_k))

        attention_heads = self.attention(q_, k_, v_)

        # (batch, num_heads, seq, d_k) ---> (batch, seq, d_model)
        h = attention_heads.view(attention_heads.shape[0], attention_heads.shape[2], attention_heads.shape[1]*attention_heads.shape[3])

        return h @ self.w_o


# we = WordEmbeddings(512, 24)
# pe = PositionalEmbeddings(512, 10)
# ff = FeedForward(4, 20)
# rs = ResidualConnection(4)
# ma = MultiHeadAttention(512, 8)

# s = "I wonder what will come next"
# tokens = torch.LongTensor([[11, 23, 21, 22, 5, 15]])

# word_embed = we(tokens)
# pos_embed = pe(word_embed)
# post_ma = ma(pos_embed, pos_embed, pos_embed)


# print(post_rs)