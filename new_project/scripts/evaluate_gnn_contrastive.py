import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARQUET_FILE = "processed/curated_subset_named.parquet"
MODEL_FILE = "models/gnn_contrastive.pt"


# ============================================================
# IMPORTANT
# MUST MATCH TRAINING SCRIPT EXACTLY
# ============================================================

ATOM_LIST = [1, 6, 7, 8, 16, 17]


def atom_features(atom):

    atomic_num = atom.GetAtomicNum()

    one_hot = [0] * len(ATOM_LIST)

    if atomic_num in ATOM_LIST:
        one_hot[ATOM_LIST.index(atomic_num)] = 1

    return one_hot


def smiles_to_graph(smiles):

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

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


def lorentzian(x, x0, gamma):
    return (gamma**2) / ((x - x0)**2 + gamma**2)


def build_spectrum(freqs, intensities,
                   wn_min=400,
                   wn_max=4000,
                   n_points=1024,
                   gamma=12):

    x_axis = np.linspace(wn_min, wn_max, n_points)

    spectrum = np.zeros_like(x_axis)

    for f, inten in zip(freqs, intensities):

        if f < wn_min or f > wn_max:
            continue

        spectrum += inten * lorentzian(x_axis, f, gamma)

    if spectrum.max() > 0:
        spectrum = spectrum / spectrum.max()

    return spectrum.astype(np.float32)


print("Loading dataset...")

df = pd.read_parquet(PARQUET_FILE)

graphs = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    graph = smiles_to_graph(row["smiles"])

    if graph is None:
        continue

    spectrum = build_spectrum(
        row["freq"],
        row["raman_activity"]
    )

    graph.spectrum = torch.tensor(spectrum, dtype=torch.float)

    graphs.append(graph)

print(f"Loaded {len(graphs)} molecules")


# ============================================================
# EXACT TRAINING ARCHITECTURE
# ============================================================

class GNNEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(6, 128),
                nn.ReLU(),
                nn.Linear(128, 128)
            )
        )

        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128)
            )
        )

        self.conv3 = GINConv(
            nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128)
            )
        )

        self.fc = nn.Linear(128, 256)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.fc(x)

        x = F.normalize(x, dim=1)

        return x


class SpectrumEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv1d(1, 32, kernel_size=7, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=7, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 256)
        )

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.conv(x)

        x = x.squeeze(-1)

        x = self.fc(x)

        x = F.normalize(x, dim=1)

        return x



print("Loading model...")

checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)

gnn_encoder = GNNEncoder().to(DEVICE)
spec_encoder = SpectrumEncoder().to(DEVICE)

gnn_encoder.load_state_dict(checkpoint["gnn_encoder"])
spec_encoder.load_state_dict(checkpoint["spectrum_encoder"])

gnn_encoder.eval()
spec_encoder.eval()

loader = DataLoader(graphs, batch_size=64, shuffle=False)

all_graph_emb = []
all_spec_emb = []

print("Generating embeddings...")

with torch.no_grad():

    for batch in tqdm(loader):

        batch = batch.to(DEVICE)

        graph_emb = gnn_encoder(batch)

        spectra = batch.spectrum.view(batch.num_graphs, -1).to(DEVICE)

        spec_emb = spec_encoder(spectra)

        all_graph_emb.append(graph_emb.cpu())
        all_spec_emb.append(spec_emb.cpu())

graph_emb = torch.cat(all_graph_emb, dim=0)
spec_emb = torch.cat(all_spec_emb, dim=0)

similarity = torch.matmul(spec_emb, graph_emb.T)

print("\nSimilarity matrix shape:", similarity.shape)

print("\nDiagonal similarities:")
for i in range(5):
    print(i, similarity[i, i].item())

print("\nMax similarities:")
for i in range(5):
    print(i, similarity[i].max().item())

print("\nTop matches:")
for i in range(5):
    top = torch.argmax(similarity[i]).item()
    print(f"query {i} -> match {top}")
print("\nTop 10 retrieved indices for first 5 queries:")

for i in range(5):

    ranking = torch.argsort(
        similarity[i],
        descending=True
    )

    print(
        f"query {i}:",
        ranking[:10].tolist()
    )

print("\nRank of the correct molecule:")

for i in range(5):

    ranking = torch.argsort(
        similarity[i],
        descending=True
    )

    rank = (ranking == i).nonzero(as_tuple=True)[0]

    print(
        f"query {i}:",
        rank.item()
    )

top1 = 0
top5 = 0
top10 = 0

N = similarity.shape[0]

for i in range(N):

    sims = similarity[i]

    ranking = torch.argsort(sims, descending=True)

    ranking = ranking.cpu().numpy()

    if i == ranking[0]:
        top1 += 1

    if i in ranking[:5]:
        top5 += 1

    if i in ranking[:10]:
        top10 += 1

print("\nRESULTS\n")

print(f"Top-1 Accuracy : {top1 / N:.4f}")
print(f"Top-5 Accuracy : {top5 / N:.4f}")
print(f"Top-10 Accuracy: {top10 / N:.4f}")
