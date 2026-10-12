import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool

from rdkit import Chem

# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "processed/curated_subset_named.parquet"

MODEL_PATH = "models/gnn_contrastive_old.pt"

TEST_INDICES = "processed/test_indices.npy"

EMBED_DIM = 256
NUM_POINTS = 1024

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# SPECTRUM
# ============================================================

def lorentzian(x, x0, gamma):
    return (gamma**2) / ((x - x0)**2 + gamma**2)

def make_spectrum(
    freqs,
    intensities,
    min_w=400,
    max_w=4000,
    num_points=NUM_POINTS,
    gamma=12
):

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

ATOM_LIST = [1, 6, 7, 8, 16, 17]

def atom_features(atom):

    feat = np.zeros(6)

    z = atom.GetAtomicNum()

    if z in ATOM_LIST:
        feat[ATOM_LIST.index(z)] = 1

    return feat

def mol_to_graph(smiles):

    mol = Chem.MolFromSmiles(smiles)

    x = []

    for atom in mol.GetAtoms():
        x.append(atom_features(atom))

    x = torch.tensor(
        x,
        dtype=torch.float
    )

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
# GNN ENCODER
# ============================================================

class GNNEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        nn1 = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        nn2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        nn3 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.conv3 = GINConv(nn3)

        self.fc = nn.Linear(
            128,
            256
        )

    def forward(self, data):

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

        self.net = nn.Sequential(

            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 256)

        )

    def forward(self, x):

        x = self.net(x)

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

print("Loading model...")

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
    ckpt["spectrum_encoder"]
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

print("Computing similarity matrix...")

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
