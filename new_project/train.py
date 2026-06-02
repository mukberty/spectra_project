import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from utils.dataset import SpectraDataset
from models.spectra_cnn import SpectraCNN


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)

dataset = SpectraDataset()

train_idx = np.load(
    "data/processed/train_indices.npy"
)

test_idx = np.load(
    "data/processed/test_indices.npy"
)

train_dataset = Subset(
    dataset,
    train_idx
)

test_dataset = Subset(
    dataset,
    test_idx
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=8
)

model = SpectraCNN().to(device)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=dataset.pos_weight.to(device)
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

for epoch in range(30):

    model.train()

    running_loss = 0

    for X,Y in train_loader:

        X = X.to(device)
        Y = Y.to(device)

        logits,_ = model(X)

        loss = criterion(
            logits,
            Y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"epoch {epoch+1}: {running_loss/len(train_loader):.4f}"
    )

torch.save(
    model.state_dict(),
    "checkpoints/spectra_classifier.pt"
)
