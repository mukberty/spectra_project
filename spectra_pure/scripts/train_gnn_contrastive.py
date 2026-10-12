# scripts/train_gnn_contrastive.py

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from rdkit import Chem

# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "processed/curated_subset_named.parquet"

SAVE_PATH = "models/gnn_contrastive.pt"

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3

EMBED_DIM = 128

NUM_POINTS = 2048

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...\n")

df = pd.read_parquet(DATA_PATH)

print(f"Total molecules: {len(df)}")

# ============================================================
# SPECTRUM GENERATION
# ============================================================

def lorentzian(x, x0, gamma):
    return (gamma**2) / ((x - x0)**2 + gamma**2)

def make_spectrum(freqs, intensities,
                  min_w=200,
                  max_w=3500,
                  num_points=NUM_POINTS,
                  gamma=12):

    x = np.linspace(min_w, max_w, num_points)

    y = np.zeros_like(x)

    for f, inten in zip(freqs, intensities):

        if f < min_w or f > max_w:
            continue

        y += inten * lorentzian(x, f, gamma)

    y = y / (y.max() + 1e-8)

    return y.astype(np.float32)

# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment_spectrum(spec):

    spec = spec.copy()

    # noise
    noise = np.random.normal(0, 0.01, spec.shape)

    spec += noise

    # intensity scaling
    scale = np.random.uniform(0.9, 1.1)

    spec *= scale

    # clipping
    spec = np.clip(spec, 0, None)

    # renormalize
    spec = spec / (spec.max() + 1e-8)

    return spec.astype(np.float32)

# ============================================================
# GRAPH CONVERSION
# ============================================================

ATOM_MAP = {
    1: 0,
    6: 1,
    7: 2,
    8: 3,
    9: 4,
    15: 5,
    16: 6,
    17: 7,
    35: 8,
    53: 9
}

def atom_features(atom):

    z = atom.GetAtomicNum()

    idx = ATOM_MAP.get(z, 10)

    feat = np.zeros(11)

    feat[idx] = 1

    return feat

def mol_to_graph(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    x = []

    for atom in mol.GetAtoms():
        x.append(atom_features(atom))

    x = torch.tensor(x, dtype=torch.float)

    edge_index = []

    for bond in mol.GetBonds():

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_index.append([i, j])
        edge_index.append([j, i])

    edge_index = torch.tensor(edge_index,
                              dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)

    return data

# ============================================================
# BUILD DATASET
# ============================================================

print("\nPreparing dataset...\n")

dataset = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    try:

        freqs = row["freq"]
        intensities = row["raman_activity"]

        smiles = row["smiles"]

        graph = mol_to_graph(smiles)

        if graph is None:
            continue

        spectrum = make_spectrum(freqs, intensities)

        dataset.append({
            "graph": graph,
            "spectrum": spectrum,
            "smiles": smiles
        })

    except:
        continue

print(f"\nFinal dataset size: {len(dataset)}")

# ============================================================
# SPLIT TRAIN / TEST
# ============================================================

train_idx = np.load(
    "processed/train_indices.npy"
)

test_idx = np.load(
    "processed/test_indices.npy"
)

train_dataset = [
    dataset[i]
    for i in train_idx
]

test_dataset = [
    dataset[i]
    for i in test_idx
]

# ============================================================
# DATALOADER
# ============================================================

class RamanDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        spec = augment_spectrum(item["spectrum"])

        return (
            item["graph"],
            torch.tensor(spec, dtype=torch.float32)
        )

def collate_fn(batch):

    graphs = [x[0] for x in batch]

    spectra = torch.stack([x[1] for x in batch])

    from torch_geometric.data import Batch

    graph_batch = Batch.from_data_list(graphs)

    return graph_batch, spectra

train_loader = DataLoader(
    RamanDataset(train_dataset),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# ============================================================
# GNN ENCODER
# ============================================================

class GNNEncoder(nn.Module):

    def __init__(self, emb_dim=EMBED_DIM):

        super().__init__()

        self.conv1 = GCNConv(11, 128)
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, 256)

        self.fc = nn.Linear(256, emb_dim)

    def forward(self, data):

        x, edge_index, batch = (
            data.x,
            data.edge_index,
            data.batch
        )

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.fc(x)

        return F.normalize(x, dim=1)

# ============================================================
# CNN SPECTRUM ENCODER
# ============================================================

class SpectrumEncoder(nn.Module):

    def __init__(self, embedding_dim=EMBED_DIM):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.conv(x)

        x = x.squeeze(-1)

        x = self.fc(x)

        return F.normalize(x, dim=1)

# ============================================================
# MODELS
# ============================================================

gnn_encoder = GNNEncoder().to(DEVICE)

spec_encoder = SpectrumEncoder().to(DEVICE)

params = (
    list(gnn_encoder.parameters()) +
    list(spec_encoder.parameters())
)

optimizer = torch.optim.Adam(params, lr=LR)

# ============================================================
# HARD NEGATIVE CONTRASTIVE LOSS
# ============================================================

def contrastive_loss(spec_emb, mol_emb, temperature=0.07):

    logits = torch.matmul(spec_emb, mol_emb.T)

    logits = logits / temperature

    batch_size = spec_emb.shape[0]

    labels = torch.arange(batch_size).to(DEVICE)

    # harder negatives
    mask = torch.eye(batch_size).bool().to(DEVICE)

    negatives = logits.masked_fill(mask, -1e9)

    hardest_negatives, _ = negatives.max(dim=1)

    positive_logits = logits[range(batch_size), labels]

    triplet_margin = 0.2

    triplet_loss = F.relu(
        hardest_negatives - positive_logits + triplet_margin
    ).mean()

    ce_loss_1 = F.cross_entropy(logits, labels)

    ce_loss_2 = F.cross_entropy(logits.T, labels)

    return (
        ce_loss_1 +
        ce_loss_2 +
        0.5 * triplet_loss
    )

# ============================================================
# TRAINING
# ============================================================

print("\nTraining contrastive GNN model...\n")

for epoch in range(EPOCHS):

    gnn_encoder.train()
    spec_encoder.train()

    total_loss = 0

    for graphs, spectra in train_loader:

        graphs = graphs.to(DEVICE)
        spectra = spectra.to(DEVICE)

        optimizer.zero_grad()

        mol_emb = gnn_encoder(graphs)

        spec_emb = spec_encoder(spectra)

        loss = contrastive_loss(spec_emb, mol_emb)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

torch.save(
    {
        "spectrum_encoder": spec_encoder.state_dict(),
        "gnn_encoder": gnn_encoder.state_dict(),

        "atom_feature_dim": 11,
        "embedding_dim": 256,

        "num_points": 2048,
        "min_w": 200,
        "max_w": 3500,
        "gamma": 12
    },
    "models/gnn_contrastive_v2.pt"
)

print("\nSaved model to:")
print(SAVE_PATH)
