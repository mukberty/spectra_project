import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from utils.dataset import SpectraDataset
from models.spectra_cnn import SpectraCNN

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)

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

model = SpectraCNN().to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/spectra_classifier.pt"
    )
)

model.eval()

all_preds = []
all_targets = []

with torch.no_grad():

    for X, Y in loader:

        X = X.to(device)

        logits, _ = model(X)

        preds = (
            torch.sigmoid(logits) > 0.5
        ).cpu().numpy()

        all_preds.append(preds)
        all_targets.append(Y.numpy())

all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

accuracy = (
    all_preds == all_targets
).mean()

print("Accuracy:", accuracy)

print(
    "Micro F1:",
    f1_score(
        all_targets,
        all_preds,
        average="micro"
    )
)

print(
    "Macro F1:",
    f1_score(
        all_targets,
        all_preds,
        average="macro"
    )
)

print(
    "\nPer-class report:\n"
)

print(
    classification_report(
        all_targets,
        all_preds,
	target_names=dataset.label_names,
        zero_division=0
    )
)
