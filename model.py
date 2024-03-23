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
    
    def forward(self, x) -> torch.Tensor:
        return self.embedding_table(x) * torch.sqrt(torch.tensor(self.d_model))

class PositionalEmbeddings(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        p_embed_table = torch.zeros((self.max_seq_len, self.d_model))
        position = torch.arange(0, max_seq_len).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(-1 * (torch.arange(0, d_model, 2).float() * (torch.log(torch.tensor(10000.0)) / self.d_model))).unsqueeze(0) # (1, d_model)

        p_embed_table[:, 0::2] = torch.sin(position * div_term)
        p_embed_table[:, 1::2] = torch.cos(position * div_term)

        p_embed_table = p_embed_table.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('p_embed_table', p_embed_table)
    
    def forward(self, x) -> torch.Tensor:
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
    def forward(self, x) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_norm = torch.nn.LayerNorm(d_model)
    
    def forward(self, x, sublayer) -> torch.Tensor:
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
    
    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        numerator = q @ k.transpose(-2, -1)
        denominator = np.sqrt(self.d_k)
        attention = self.softmax(numerator/denominator)
        if self.mask:
            attention = attention @ self.mask
        return attention @ v

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
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

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, num_heads: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.MHA = MultiHeadAttention(self.d_model, self.num_heads)
        self.FF = FeedForward(self.d_model, self.d_hidden)
        self.RCs = nn.ModuleList([ResidualConnection(self.d_model) for _ in range(2)]) 
    
    def forward(self, x) -> torch.Tensor:
        x = self.RCs[0](x, lambda x: self.MHA(q=x, k=x, v=x))
        x = self.RCs[1](x, self.FF)
        return x

class Encoder(nn.Module):
    def __init__(self, num_blocks: int, d_model: int, d_hidden: int, num_heads: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.encoders = nn.ModuleList([EncoderBlock(self.d_model, self.d_hidden, self.num_heads) for _ in range(self.num_blocks)])

    def forward(self, x) -> torch.Tensor:
        for enc in self.encoders:
            x = enc(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, num_heads: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.MMHA = MultiHeadAttention(self.d_model, self.num_heads, mask=None)
        self.MHA = MultiHeadAttention(self.d_model, self.num_heads)
        self.FF = FeedForward(self.d_model, self.d_hidden)
        self.RCs = nn.ModuleList([ResidualConnection(self.d_model) for _ in range(3)])
    
    def forward(self, x, enc_k, enc_v) -> torch.Tensor:
        x = self.RCs[0](x, lambda x: self.MMHA(q=x, k=x, v=x))
        x = self.RCs[1](x, lambda x: self.MHA(q=x, k=enc_k, v=enc_v))
        x = self.RCs[2](x, self.FF)
        return x

class Decoder(nn.Module):
    def __init__(self, num_blocks: int, d_model: int, d_hidden: int, num_heads: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.decoders = nn.ModuleList([DecoderBlock(self.d_model, self.d_hidden, self.num_heads) for _ in range(self.num_blocks)])
    
    def forward(self, x, enc_k, enc_v) -> torch.Tensor:
        for dec in self.decoders:
            x = dec(x, enc_k, enc_v)
        return x

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (batch, seq_len, d_model) --->  (batch, seq_len, vocab_size)
        x = self.linear(x)
        return x

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, projection: ProjectionLayer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.projection = projection
    
    def encode(self):
        pass

    def decode(self):
        pass

    def project(self):
        pass


def create_model():
    pass

if __name__ == "__main__":
    d_model = 512
    vocab_size = 24
    max_seq_len = 10
    d_hidden = 2048
    num_heads = 4
    num_blocks = 6

    we = WordEmbeddings(d_model, vocab_size)
    pe = PositionalEmbeddings(d_model, max_seq_len)
    enc = Encoder(num_blocks, d_model, d_hidden, num_heads)

    s = "I wonder what will come next"
    tokens = torch.LongTensor([[11, 23, 21, 22, 5, 15]])

    word_embed = we(tokens)
    pos_embed = pe(word_embed)
    enc_output = enc(pos_embed)
    print(enc_output.shape)