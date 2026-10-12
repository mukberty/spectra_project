import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

from rdkit import Chem

# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "processed/curated_subset_named.parquet"

MODEL_PATH = "models/gnn_contrastive.pt"

TEST_INDICES = "processed/test_indices.npy"

EMBED_DIM = 128
NUM_POINTS = 2048

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# SPECTRUM
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
# GRAPHS
# ============================================================

ATOM_MAP = {
    1:0,
    6:1,
    7:2,
    8:3,
    9:4,
    15:5,
    16:6,
    17:7,
    35:8,
    53:9
}

def atom_features(atom):

    z = atom.GetAtomicNum()

    idx = ATOM_MAP.get(z, 10)

    feat = np.zeros(11)

    feat[idx] = 1

    return feat

def mol_to_graph(smiles):

    mol = Chem.MolFromSmiles(smiles)

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

    edge_index = torch.tensor(
        edge_index,
        dtype=torch.long
    ).t().contiguous()

    return Data(
        x=x,
        edge_index=edge_index
    )

# ============================================================
# GNN
# ============================================================

class GNNEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = GCNConv(11,128)
        self.conv2 = GCNConv(128,256)
        self.conv3 = GCNConv(256,256)

        self.fc = nn.Linear(
            256,
            EMBED_DIM
        )

    def forward(self,data):

        x = self.conv1(
            data.x,
            data.edge_index
        )
        x = F.relu(x)

        x = self.conv2(
            x,
            data.edge_index
        )
        x = F.relu(x)

        x = self.conv3(
            x,
            data.edge_index
        )
        x = F.relu(x)

        batch = torch.zeros(
            len(x),
            dtype=torch.long,
            device=x.device
        )

        x = global_mean_pool(
            x,
            batch
        )

        x = self.fc(x)

        return F.normalize(
            x,
            dim=1
        )

# ============================================================
# SPECTRUM ENCODER
# ============================================================

class SpectrumEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv1d(
                1,32,
                kernel_size=7,
                padding=3
            ),
            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                32,64,
                kernel_size=7,
                padding=3
            ),
            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                64,128,
                kernel_size=7,
                padding=3
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(
            128,
            EMBED_DIM
        )

    def forward(self,x):

        x = x.unsqueeze(1)

        x = self.conv(x)

        x = x.squeeze(-1)

        x = self.fc(x)

        return F.normalize(
            x,
            dim=1
        )

# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_parquet(DATA_PATH)

test_idx = np.load(TEST_INDICES)

df = df.iloc[test_idx].reset_index(drop=True)

print("Test molecules:", len(df))

# ============================================================
# LOAD MODEL
# ============================================================

gnn = GNNEncoder().to(DEVICE)
spec = SpectrumEncoder().to(DEVICE)

ckpt = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

gnn.load_state_dict(
    ckpt["gnn_encoder"]
)

spec.load_state_dict(
    ckpt["spec_encoder"]
)

gnn.eval()
spec.eval()

# ============================================================
# EMBEDDINGS
# ============================================================

print("Computing embeddings...")

graph_embs = []
spec_embs = []

with torch.no_grad():

    for _, row in df.iterrows():

        graph = mol_to_graph(
            row["smiles"]
        )

        graph = graph.to(DEVICE)

        g = gnn(graph)

        spectrum = make_spectrum(
            row["freq"],
            row["raman_activity"]
        )

        spectrum = torch.tensor(
            spectrum,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        s = spec(spectrum)

        graph_embs.append(
            g.cpu()
        )

        spec_embs.append(
            s.cpu()
        )

graph_embs = torch.cat(graph_embs)
spec_embs = torch.cat(spec_embs)

# ============================================================
# RETRIEVAL
# ============================================================

similarity = torch.matmul(
    spec_embs,
    graph_embs.T
)

top1 = 0
top5 = 0
top10 = 0

N = len(df)

for i in range(N):

    ranking = torch.argsort(
        similarity[i],
        descending=True
    )

    if i == ranking[0]:
        top1 += 1

    if i in ranking[:5]:
        top5 += 1

    if i in ranking[:10]:
        top10 += 1

print()
print("RESULTS")
print()

print(
    f"Top-1 Accuracy : {top1/N:.4f}"
)

print(
    f"Top-5 Accuracy : {top5/N:.4f}"
)

print(
    f"Top-10 Accuracy: {top10/N:.4f}"
)
