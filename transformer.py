import math

import torch
import torch.nn as nn


# A dictionary for vocabulary embeddings.
# shape in papers: D x |V|
# shape in code: |V| x D
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)


# Adds positional information to a sequence of tokens
# PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) ; even dimention
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)); odd dimention
# but to implement it, we will do exp and log
# #FAFO << write detail reason why>>
# PE(pos,2i) = sin(pos/exp(log(10000^(2i/d_model))))
#            = sin(pos/exp((2i/d_model)*log(10000)))
#            = sin(pos*exp(-1*(2i/d_model)*log(10000)))
class PositionalEncodings(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # position embeddings for seq_len
        pe = torch.zeros(seq_len, d_model)

        positions = torch.arange(0, self.seq_len, dtype=torch.float)
        # FAFO
        # what does unsqueeze mean? and why is it needed here?
        positions = positions.unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            / self.d_model
            * -1
            * math.log(10000)
        )
        # Even positions: 0,2,4
        pe[:, 0::2] = torch.sin(positions * div_term)
        # Odd positions: 1,3,5
        pe[:, 1::2] = torch.cos(positions * div_term)

        # now we will add the Batch dimention
        # FAFO
        # this somehow adds a new dimention
        pe = pe.unsqueeze(0)  # (1, Seq_len, d_model)

        # added so that linter doesn't complain
        self.pe: torch.Tensor
        self.register_buffer("pe", pe)

    # x is the batch of sequences, so we would want to add the whole pe to x
    # but some sequence in the match may not be of size seq_len, so we pick up
    # only that much from pe as is there in the sequence
    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return x


# each vector needs to be normalized
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # multipliocative
        self.bias = nn.Parameter(torch.zeros(1))  # additive

    def forward(self, x: torch.Tensor):
        # find mean of the whole Batch, but which columns???
        # find variance similarly
        # dim=-1 means we are doing average of the vector values in the last dimention
        # keepdim=True means we want to retain the dimention which got reduced
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(in_features=d_ff, out_features=d_model)

    def forward(self, x):
        # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_ff) -> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model: {} is not divisible by h: {}".format(
            d_model, h
        )
        self.d_k = d_model // h
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_o = nn.Linear(in_features=d_model, out_features=d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask,
        dropout: nn.Dropout,
    ):
        d_k = query.shape[-1]
        # query: (Batch, h, Seq_Len, d_k)
        # key: (Batch, h, Seq_Len, d_k)
        # key transpose = (Batch, h, d_k, Seq_len)
        # attn_scores = (Batch, h, Seq_Len, Seq_Len)
        attn_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            attn_scores.masked_fill_(mask == 0, -1e9)
        # (Batch, h, Seq_Len, Seq_Len)
        attn_scores = attn_scores.softmax(dim=-1)
        if dropout is not None:
            attn_scores = dropout(attn_scores)
        # value: (Batch, h, Seq_Len, d_k)

        # retured marix shape: (Batch, h, Seq_Len, d_k)
        return (attn_scores @ value), attn_scores

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask):
        query: torch.Tensor
        query = self.w_q(q)  # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_model)
        key: torch.Tensor
        key = self.w_k(k)  # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_model)
        value: torch.Tensor
        value = self.w_v(v)  # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_model)

        # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len,h, d_k) -> (Batch, h, Seq_Len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        x, self.attn_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self, self_attn: MultiHeadAttentionBlock, ff: FeedForwardBlock, dropout: float
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.ff = ff
        self.res1 = ResidualConnection(dropout)
        self.res2 = ResidualConnection(dropout)

    def forward(self, x: torch.Tensor, mask):
        x = self.res1(x, lambda y: self.self_attn(y, y, y, mask))
        x = self.res2(x, self.ff)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.norm = LayerNormalization()
        self.layers = layers

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attn: MultiHeadAttentionBlock,
        cross_attn: MultiHeadAttentionBlock,
        ff: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ff = ff
        self.dropout = nn.Dropout(dropout)
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(self, x, enc_x, enc_mask, dec_mask):
        x = self.residual_connection[0](x, lambda y: self.self_attn(y, y, y, dec_mask))
        x = self.residual_connection[1](
            x, lambda y: self.cross_attn(y, enc_x, enc_x, enc_mask)
        )
        x = self.residual_connection[2](x, self.ff)
        return self.dropout(x)


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.norm = LayerNormalization()
        self.layers = layers

    def forward(self, x, encoder_x, enc_mask, dec_mask):
        for layer in self.layers:
            x = layer(x, encoder_x, enc_mask, dec_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x):
        return self.linear(torch.log_softmax(x, dim=-1))


class Transformer(nn.Module):
    def __init__(
        self,
        enc_embed: InputEmbeddings,
        encoder_pe: PositionalEncodings,
        encoder: Encoder,
        dec_embed: InputEmbeddings,
        decoder_pe: PositionalEncodings,
        decoder: Decoder,
        projection: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.enc_embed = enc_embed
        self.encoder_pe = encoder_pe
        self.encoder = encoder
        self.dec_embed = dec_embed
        self.decoder_pe = decoder_pe
        self.decoder = decoder
        self.projection = projection

    def encode(self, src, src_mask):
        src = self.enc_embed(src)
        src = self.encoder_pe(src)
        return self.encoder(src, src_mask)

    def decode(self, x, encoder_x, enc_mask, dec_mask):
        x = self.dec_embed(x)
        x = self.decoder_pe(x)
        return self.decoder(x, encoder_x, enc_mask, dec_mask)

    def project(self, x):
        return self.projection(x)


def build_transformer(
    src_vocab_size: int,
    target_vocab_size: int,
    src_seq_len: int,
    target_seq_len: int,
    d_model: int = 512,
    N: int = 6,  # Number of Blocks/Layers
    h: int = 8,  # Number of heads
    dropout: float = 0.1,
    d_ff: int = 2048,
):
    # Embedding Layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    target_embed = InputEmbeddings(d_model, target_vocab_size)

    # Positional Encoding Layers
    src_pe = PositionalEncodings(src_seq_len, d_model, dropout)
    target_pe = PositionalEncodings(target_seq_len, d_model, dropout)

    # Encoder Layers
    encoder_blocks = []
    for _ in range(N):
        b = EncoderBlock(
            MultiHeadAttentionBlock(d_model, h, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            dropout,
        )
        encoder_blocks.append(b)
    # Encoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    # Decoder Layers
    decoder_blocks = []
    for _ in range(N):
        b = DecoderBlock(
            MultiHeadAttentionBlock(d_model, h, dropout),
            MultiHeadAttentionBlock(d_model, h, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            dropout,
        )
        decoder_blocks.append(b)
    # Decoder
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Projection layer for target vocabulary
    projection = ProjectionLayer(d_model, target_vocab_size)

    # finally, the transformer
    transformer = Transformer(
        src_embed,
        src_pe,
        encoder,
        target_embed,
        target_pe,
        decoder,
        projection,
    )
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer


if __name__ == "__main__":
    t = build_transformer(
        src_vocab_size=20000,
        target_vocab_size=20000,
        src_seq_len=200,
        target_seq_len=200,
    )
    iter_loop = 0
    sum_params = 0
    for p in t.parameters():
        iter_loop += 1
        sum_params += p.numel()
    print(
        "transformer number of parameters. :iter_loop: {}, sum_param: {}".format(
            iter_loop, sum_params
        )
    )
