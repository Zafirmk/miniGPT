import math
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
        return self.embedding_table(x) * math.sqrt(self.d_model)

class PositionalEmbeddings(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

        p_embed_table = torch.zeros((self.max_seq_len, self.d_model))
        position = torch.arange(0, max_seq_len).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(-1 * (torch.arange(0, d_model, 2).float() * (torch.log(torch.tensor(10000.0)) / self.d_model))).unsqueeze(0) # (1, d_model)

        p_embed_table[:, 0::2] = torch.sin(position * div_term)
        p_embed_table[:, 1::2] = torch.cos(position * div_term)

        p_embed_table = p_embed_table.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('p_embed_table', p_embed_table)
    
    def forward(self, x) -> torch.Tensor:
        with torch.no_grad():
            return self.dropout(x + self.p_embed_table[:, :x.shape[1], :])

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = torch.nn.Linear(d_model, d_hidden)
        self.linear2 = torch.nn.Linear(d_hidden, d_model)
    
    # (batch, seq_len, d_model) --> (batch, seq_len, d_hidden) --> (batch, seq_len, d_model)
    def forward(self, x) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class LayerNormalization(nn.Module):
    def __init__(self, in_features: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = 10**-6
        self.a = nn.Parameter(torch.ones(in_features))
        self.b = nn.Parameter(torch.zeros(in_features))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.a * (x - mean) / (std + self.eps) + self.b

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)
    
    def forward(self, x, sublayer) -> torch.Tensor:
        return x + self.dropout(sublayer(self.layer_norm(x)))

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.softmax = torch.nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.w_q = torch.nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = torch.nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = torch.nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = torch.nn.Linear(self.d_model, self.d_model, bias=False)
    
    
    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout: float, padding_mask: torch.Tensor = None) -> torch.Tensor:
        numerator = q @ k.transpose(-2, -1)
        denominator = math.sqrt(self.d_k)
        attn_logits = (numerator / denominator)

        if padding_mask is not None:
            attn_logits.masked_fill_(padding_mask == 0, -1e9)

        attn_probs = self.softmax(attn_logits)
        
        if dropout is not None:
            attn_probs = dropout(attn_probs)

        out = attn_probs @ v

        # attn_probs = [batch, heads, seq_len, seq_len]
        return out

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        q_ = self.w_q(q)
        k_ = self.w_k(k)
        v_ = self.w_v(v)

        q_ = q_.view((q_.shape[0], q_.shape[1], self.num_heads, int(self.d_k))).transpose(1, 2).type(dtype=torch.float32)
        k_ = k_.view((k_.shape[0], k_.shape[1], self.num_heads, int(self.d_k))).transpose(1, 2).type(dtype=torch.float32)
        v_ = v_.view((v_.shape[0], v_.shape[1], self.num_heads, int(self.d_k))).transpose(1, 2).type(dtype=torch.float32)

        attn_heads = self.attention(q_, k_, v_, padding_mask, self.dropout)
        attn_heads = attn_heads.transpose(1, 2).contiguous().view(attn_heads.shape[0], -1, int(self.num_heads * self.d_k))

        out = self.w_o(attn_heads)

        return out

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, num_heads: int, dropout: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.MHA = MultiHeadAttention(self.d_model, self.num_heads, dropout)
        self.FF = FeedForward(self.d_model, self.d_hidden, dropout)
        self.RCs = nn.ModuleList([ResidualConnection(self.d_model, dropout) for _ in range(2)]) 
    
    def forward(self, x, enc_padding_mask) -> torch.Tensor:
        x = self.RCs[0](x, lambda x: self.MHA(q=x, k=x, v=x, padding_mask=enc_padding_mask))
        x = self.RCs[1](x, self.FF)
        return x

class Encoder(nn.Module):
    def __init__(self, num_blocks: int, d_model: int, d_hidden: int, num_heads: int, dropout: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.encoders = nn.ModuleList([EncoderBlock(self.d_model, self.d_hidden, self.num_heads, dropout) for _ in range(self.num_blocks)])
        self.Lnorm = LayerNormalization(self.d_model)

    def forward(self, x, mask) -> torch.Tensor:
        for enc in self.encoders:
            x = enc(x, mask)
        return self.Lnorm(x)

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, num_heads: int, dropout: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.MMHA = MultiHeadAttention(self.d_model, self.num_heads, dropout)
        self.MHA = MultiHeadAttention(self.d_model, self.num_heads, dropout)
        self.FF = FeedForward(self.d_model, self.d_hidden, dropout)
        self.RCs = nn.ModuleList([ResidualConnection(self.d_model, dropout) for _ in range(3)])
    
    def forward(self, x, enc_k, enc_v, enc_padding_mask, dec_padding_mask) -> torch.Tensor:
        x = self.RCs[0](x, lambda x: self.MMHA(q=x, k=x, v=x, padding_mask=dec_padding_mask))
        x = self.RCs[1](x, lambda x: self.MHA(q=x, k=enc_k, v=enc_v, padding_mask=enc_padding_mask))
        x = self.RCs[2](x, self.FF)
        return x

class Decoder(nn.Module):
    def __init__(self, num_blocks: int, d_model: int, d_hidden: int, num_heads: int, dropout: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.decoders = nn.ModuleList([DecoderBlock(self.d_model, self.d_hidden, self.num_heads, dropout) for _ in range(self.num_blocks)])
        self.Lnorm = LayerNormalization(self.d_model)
    
    def forward(self, x, enc_k, enc_v, enc_padding_mask, dec_padding_mask) -> torch.Tensor:
        for dec in self.decoders:
            x = dec(x, enc_k, enc_v, enc_padding_mask, dec_padding_mask)
        return self.Lnorm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (batch, seq_len, d_model) --->  (batch, seq_len, vocab_size)
        x = self.linear(x)
        return x

class EncoderDecoderTransformer(nn.Module):
    def __init__(self, num_blocks: int, num_heads: int, d_model: int, d_hidden: int, enc_vocab_size: int, dec_vocab_size: int, max_seq_len: int, dropout: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = Encoder(num_blocks, d_model, d_hidden, num_heads, dropout)
        self.decoder = Decoder(num_blocks, d_model, d_hidden, num_heads, dropout)
        self.projection = ProjectionLayer(d_model, dec_vocab_size)
        self.enc_vocab = WordEmbeddings(d_model, enc_vocab_size)
        self.dec_vocab = WordEmbeddings(d_model, dec_vocab_size)
        self.positional_enc = PositionalEmbeddings(d_model, max_seq_len, dropout)
    
    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.enc_vocab(x)
        x = self.positional_enc(x)
        x = self.encoder(x, mask)
        return x

    def decode(self, x: torch.Tensor, enc_k: torch.Tensor, enc_v: torch.Tensor, enc_padding_mask: torch.Tensor, dec_padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.dec_vocab(x)
        x = self.positional_enc(x)
        x = self.decoder(x, enc_k, enc_v, enc_padding_mask, dec_padding_mask)
        return x

    def project(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        return x
    
    def forward(self, enc_input: torch.Tensor, dec_input: torch.Tensor, enc_padding_mask: torch.Tensor, dec_padding_mask: torch.Tensor) -> torch.Tensor:
        enc_output = self.encode(enc_input, enc_padding_mask)
        dec_output = self.decode(dec_input, enc_output, enc_output, enc_padding_mask, dec_padding_mask)
        return self.project(dec_output)


def create_model(model_type: EncoderDecoderTransformer, **kwargs) -> EncoderDecoderTransformer:
    model = model_type(**kwargs)
    flag = False
    for p in model.parameters():
        flag = True
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    if flag:
        print("Model weights initialized with Xavier Uniform")
    return model
