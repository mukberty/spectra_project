import torch
import torch.nn as nn


class SmilesEncoder(nn.Module):

    def __init__(self, vocab_size, embed_dim=256, max_len=128):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )

        self.proj = nn.Linear(embed_dim, embed_dim)

        self.max_len = max_len

    def forward(self, x):
        """
        x: (B, L) tokenized SMILES
        """

        B, L = x.shape
        L = min(L, self.max_len)

        x = x[:, :L]

        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)

        h = self.token_emb(x) + self.pos_emb(pos)

        h = self.encoder(h)

        h = h.mean(dim=1)

        return self.proj(h)
