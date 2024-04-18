import os
import torch
import torch.nn as nn

class WordEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding_table = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model,
        )
    
    def forward(self, x) -> torch.Tensor:
        temp = self.embedding_table(x)
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
    def __init__(self, d_model: int, num_heads: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model / num_heads
        self.softmax = torch.nn.Softmax(dim=-1)
        self.d_model = d_model
        self.num_heads = num_heads

        self.w_q = torch.randn((self.d_model, self.d_model), dtype=torch.float32).to(f'cuda:{(os.environ["LOCAL_RANK"])}')
        self.w_k = torch.randn((self.d_model, self.d_model), dtype=torch.float32).to(f'cuda:{(os.environ["LOCAL_RANK"])}')
        self.w_v = torch.randn((self.d_model, self.d_model), dtype=torch.float32).to(f'cuda:{(os.environ["LOCAL_RANK"])}')
        self.w_o = torch.randn((self.d_model, self.d_model), dtype=torch.float32).to(f'cuda:{(os.environ["LOCAL_RANK"])}')
    
    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        numerator = q @ k.transpose(-2, -1)
        denominator = torch.sqrt(torch.tensor(self.d_k, dtype=q.dtype))
        # (batch, num_heads, seq_len, seq_len)
        attention_scores = numerator / denominator
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_probs = self.softmax(attention_scores)
        output = attention_probs @ v
        return output

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q_ = q @ self.w_q
        k_ = k @ self.w_k
        v_ = v @ self.w_v

        # (batch, seq, d_model) ---> (batch, num_heads, seq, d_k)
        q_ = q_.view(q_.shape[0], self.num_heads, q.shape[1], int(self.d_k)).type(dtype=torch.float32)
        k_ = k_.view(k_.shape[0], self.num_heads, k.shape[1], int(self.d_k)).type(dtype=torch.float32)
        v_ = v_.view(v_.shape[0], self.num_heads, v.shape[1], int(self.d_k)).type(dtype=torch.float32)

        attention_heads = self.attention(q_, k_, v_, mask)

        # (batch, num_heads, seq, d_k) ---> (batch, seq, d_model)
        h = attention_heads.view(attention_heads.shape[0], attention_heads.shape[2], attention_heads.shape[1]*attention_heads.shape[3]).type(dtype=torch.float32)

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
    
    def forward(self, x, mask) -> torch.Tensor:
        x = self.RCs[0](x, lambda x: self.MHA(q=x, k=x, v=x, mask=mask))
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

    def forward(self, x, mask) -> torch.Tensor:
        for enc in self.encoders:
            x = enc(x, mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, num_heads: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.MMHA = MultiHeadAttention(self.d_model, self.num_heads)
        self.MHA = MultiHeadAttention(self.d_model, self.num_heads)
        self.FF = FeedForward(self.d_model, self.d_hidden)
        self.RCs = nn.ModuleList([ResidualConnection(self.d_model) for _ in range(3)])
    
    def forward(self, x, enc_k, enc_v, enc_mask, dec_mask) -> torch.Tensor:
        x = self.RCs[0](x, lambda x: self.MMHA(q=x, k=x, v=x, mask=dec_mask))
        x = self.RCs[1](x, lambda x: self.MHA(q=x, k=enc_k, v=enc_v, mask=enc_mask))
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
    
    def forward(self, x, enc_k, enc_v, enc_mask, dec_mask) -> torch.Tensor:
        for dec in self.decoders:
            x = dec(x, enc_k, enc_v, enc_mask, dec_mask)
        return x

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (batch, seq_len, d_model) --->  (batch, seq_len, vocab_size)
        x = self.linear(x)
        return x

class EncoderDecoderTransformer(nn.Module):
    def __init__(self, num_blocks: int, num_heads: int, d_model: int, d_hidden: int, enc_vocab_size: int, dec_vocab_size: int, max_seq_len: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = Encoder(num_blocks, d_model, d_hidden, num_heads)
        self.decoder = Decoder(num_blocks, d_model, d_hidden, num_heads)
        self.projection = ProjectionLayer(d_model, dec_vocab_size)
        self.enc_vocab = WordEmbeddings(d_model, enc_vocab_size)
        self.dec_vocab = WordEmbeddings(d_model, dec_vocab_size)
        self.positional_enc = PositionalEmbeddings(d_model, max_seq_len)
    
    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.enc_vocab(x)
        x = self.positional_enc(x)
        x = self.encoder(x, mask)
        return x

    def decode(self, x: torch.Tensor, enc_k: torch.Tensor, enc_v: torch.Tensor, enc_mask: torch.Tensor, dec_mask: torch.Tensor) -> torch.Tensor:
        x = self.dec_vocab(x)
        x = self.positional_enc(x)
        x = self.decoder(x, enc_k, enc_v, enc_mask, dec_mask)
        return x

    def project(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        return x
    
    def forward(self, enc_input: torch.Tensor, dec_input: torch.Tensor, enc_mask: torch.Tensor, dec_mask: torch.Tensor) -> torch.Tensor:
        enc_output = self.encode(enc_input, enc_mask)
        dec_output = self.decode(dec_input, enc_output, enc_output, enc_mask, dec_mask)
        return self.project(dec_output)


def create_model(model_type: EncoderDecoderTransformer, **kwargs) -> EncoderDecoderTransformer:
    model = model_type(**kwargs)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
