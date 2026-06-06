import torch
import torch.nn as nn


class SpectraCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv1d(1,32,7,padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32,64,7,padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64,128,5,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128,256,5,padding=2),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.embedding = nn.Linear(
            256,
            256
        )

        self.classifier = nn.Linear(
            256,
            20
        )

    def forward(self,x):

        x = x.unsqueeze(1)

        x = self.encoder(x)

        x = x.squeeze(-1)

        emb = self.embedding(x)

        logits = self.classifier(emb)

        return logits, emb
