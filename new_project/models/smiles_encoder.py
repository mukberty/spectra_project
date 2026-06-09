import torch
import torch.nn as nn


class SmilesEncoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        max_len=256,
        hidden_dim=256,
        num_layers=4,
        num_heads=8
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)

        self.pos_emb = nn.Parameter(
            torch.randn(1, max_len, hidden_dim) * 0.01
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):

        # x: [B, L] token ids

        x = self.token_emb(x)

        x = x + self.pos_emb[:, :x.shape[1]]

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.proj(x)

        return x
