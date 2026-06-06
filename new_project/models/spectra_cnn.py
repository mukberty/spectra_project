import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(),

            nn.Conv1d(channels, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class SpectraCNN(nn.Module):

    def __init__(self, embed_dim=256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            ResidualBlock(32),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            ResidualBlock(64),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            ResidualBlock(128)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.embedding = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.embedding(x)
        return x
