import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    modified from
    https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
    """

    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_length, embedding_dim]``
        """
        x = x + self.pe
        return self.dropout(x)


class CrossAttentionBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super(CrossAttentionBlock, self).__init__()

        self.multi_head_self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,  # Dropout on attn_output_weights
            batch_first=True,  # input is (batch, seq, feature)
        )

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(
        self,
        Q,
        K,
        V,
        position_encoding_K=None,
        position_encoding_Q=None,
    ):

        attrs = (Q, K, V)

        if position_encoding_K is not None:
            attrs = (attrs[0], position_encoding_K(attrs[1]), attrs[2])

        if position_encoding_Q is not None:
            attrs = (position_encoding_Q(attrs[0]), *attrs[1:])

        # Cross-attention where query comes from x and key/value comes from y
        output = self.multi_head_self_attention(*attrs)
        attn_output, attn_output_weights = output

        x = self.dropout(attn_output)

        # Add & Norm for cross-attention output
        x = self.layer_norm1(x + Q)

        # Feed Forward
        ff_output = self.feed_forward(x)

        # Add & Norm for feed-forward output
        x = self.layer_norm2(x + ff_output)

        return x


class TwoCrossAttention(nn.Module):

    def __init__(
        self,
        embed_dim_1,
        num_heads_1,
        dropout_rate_1,
        embed_dim_2,
        num_heads_2,
        dropout_rate_2,
    ):

        super(TwoCrossAttention, self).__init__()

        self.cross_att_1 = CrossAttentionBlock(
            embed_dim=embed_dim_1,
            num_heads=num_heads_1,
            dropout_rate=dropout_rate_1,
        )

        self.cross_att_2 = CrossAttentionBlock(
            embed_dim=embed_dim_2,
            num_heads=num_heads_2,
            dropout_rate=dropout_rate_2,
        )

    def forward(
        self,
        Q1,
        K1,
        V1,
        Q2,
        K2,
        V2,
        position_encoding_K=None,
        position_encoding_Q=None,
    ):

        x1 = self.cross_att_1(
            Q1,
            K2,
            V2,
            position_encoding_K=position_encoding_K,
            position_encoding_Q=position_encoding_Q,
        )
        x2 = self.cross_att_2(
            Q2,
            K1,
            V1,
            position_encoding_K=position_encoding_K,
            position_encoding_Q=position_encoding_Q,
        )
        return x1, x2
