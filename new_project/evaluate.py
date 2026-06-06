import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Subset

from models.spectra_cnn import SpectraCNN
from utils.dataset import SpectraDataset

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Load dataset and test subsets
dataset = SpectraDataset()
test_idx = np.load("data/processed/test_indices.npy")
test_dataset = Subset(dataset, test_idx)

loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Load model
model = SpectraCNN().to(device)
model.load_state_dict(torch.load("checkpoints/spectra_classifier.pt"))
model.eval()

# Load thresholds once outside the evaluation loop for better performance
thresholds = np.load("checkpoints/class_thresholds.npy")
thresholds = torch.tensor(thresholds, dtype=torch.float32).to(device)

all_preds = []
all_targets = []

with torch.no_grad():
    for X, Y in loader:
        X = X.to(device)
        
        logits, _ = model(X)
        
        # Apply sigmoid and compare against class-specific thresholds
        preds = (torch.sigmoid(logits) > thresholds).cpu().numpy()
        
        all_preds.append(preds)
        all_targets.append(Y.numpy())

# Combine batch results
all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

# Calculate and print metrics
accuracy = (all_preds == all_targets).mean()
print("Accuracy:", accuracy)

print(
    "Micro F1:", 
    f1_score(all_targets, all_preds, average="micro")
)
print(
    "Macro F1:", 
    f1_score(all_targets, all_preds, average="macro")
)

print("\nPer-class report:\n")
print(
    classification_report(
        all_targets,
        all_preds,
        target_names=dataset.label_names,
        zero_division=0
    )
)
