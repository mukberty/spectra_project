import numpy as np
import torch

from sklearn.metrics import f1_score

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from utils.dataset import SpectraDataset
from models.spectra_cnn import SpectraCNN


device = "cpu"

dataset = SpectraDataset()

test_idx = np.load(
    "data/processed/test_indices.npy"
)

test_dataset = Subset(
    dataset,
    test_idx
)

loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

model = SpectraCNN()

model.load_state_dict(
    torch.load(
        "checkpoints/spectra_classifier.pt",
        map_location=device
    )
)

model.eval()

all_probs = []
all_targets = []

with torch.no_grad():

    for X, Y in loader:

        logits, _ = model(X)

        probs = torch.sigmoid(
            logits
        ).numpy()

        all_probs.append(probs)
        all_targets.append(
            Y.numpy()
        )

all_probs = np.vstack(all_probs)
all_targets = np.vstack(all_targets)

best_thresholds = []

for c in range(all_targets.shape[1]):

    best_f1 = 0
    best_t = 0.5

    for t in np.arange(
        0.05,
        0.95,
        0.05
    ):

        pred = (
            all_probs[:, c] > t
        ).astype(int)

        f1 = f1_score(
            all_targets[:, c],
            pred
        )

        if f1 > best_f1:

            best_f1 = f1
            best_t = t

    best_thresholds.append(
        best_t
    )

print(best_thresholds)

np.save(
    "checkpoints/class_thresholds.npy",
    np.array(best_thresholds)
)
