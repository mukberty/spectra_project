import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool

from rdkit import Chem


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARQUET_PATH = "processed/curated_subset_named.parquet"
MODEL_PATH = "models/gnn_contrastive.pt"

WAVENUMBER_MIN = 400
WAVENUMBER_MAX = 4000
NUM_POINTS = 1024


# ============================================================
# FEATURES
# ============================================================

def atom_features(atom):

    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
        int(atom.GetHybridization())
    ]


def bond_features(bond):

    bt = bond.GetBondType()

    return [
        int(bt == Chem.rdchem.BondType.SINGLE),
        int(bt == Chem.rdchem.BondType.DOUBLE),
        int(bt == Chem.rdchem.BondType.TRIPLE),
        int(bt == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())
    ]


# ============================================================
# GRAPH
# ============================================================

def smiles_to_graph(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    x = []

    for atom in mol.GetAtoms():
        x.append(atom_features(atom))

    x = torch.tensor(x, dtype=torch.float)

    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bf = bond_features(bond)

        edge_index.append([i, j])
        edge_index.append([j, i])

        edge_attr.append(bf)
        edge_attr.append(bf)

    if len(edge_index) == 0:

        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)

    else:

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    return data


# ============================================================
# SPECTRUM
# ============================================================

def lorentzian(x, x0, gamma):

    return (gamma**2) / ((x - x0)**2 + gamma**2)


def generate_continuous_spectrum(freqs, intensities):

    x_axis = np.linspace(
        WAVENUMBER_MIN,
        WAVENUMBER_MAX,
        NUM_POINTS
    )

    spectrum = np.zeros_like(x_axis)

    gamma = 10

    for f, inten in zip(freqs, intensities):

        if f <= 0:
            continue

        spectrum += inten * lorentzian(
            x_axis,
            f,
            gamma
        )

    spectrum = spectrum / (spectrum.max() + 1e-8)

    return spectrum.astype(np.float32)


# ============================================================
# MODELS
# ============================================================

class SpectrumEncoder(nn.Module):

    def __init__(self):

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

        return x

class GNNEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        dim = 128

        nn1 = nn.Sequential(
            nn.Linear(6, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        nn2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        nn3 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.conv3 = GINConv(nn3)

        self.fc = nn.Linear(dim, 256)

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.fc(x)

        return x


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_parquet(PARQUET_PATH)

graphs = []
spectra = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    graph = smiles_to_graph(row["smiles"])

    if graph is None:
        continue

    spectrum = generate_continuous_spectrum(
        row["freq"],
        row["raman_activity"]
    )

    graph.spectrum = torch.tensor(
        spectrum,
        dtype=torch.float
    )

    graphs.append(graph)
    spectra.append(spectrum)

loader = DataLoader(
    graphs,
    batch_size=128,
    shuffle=False
)


# ============================================================
# LOAD MODEL
# ============================================================

spec_encoder = SpectrumEncoder().to(DEVICE)
gnn_encoder = GNNEncoder().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

spec_encoder.load_state_dict(
    checkpoint["spectrum_encoder"]
)

gnn_encoder.load_state_dict(
    checkpoint["gnn_encoder"]
)

spec_encoder.eval()
gnn_encoder.eval()


# ============================================================
# EMBEDDINGS
# ============================================================

print("\nEncoding molecules...\n")

mol_embeddings = []

with torch.no_grad():

    for batch in tqdm(loader):

        batch = batch.to(DEVICE)

        emb = gnn_encoder(batch)

        emb = F.normalize(emb, dim=1)

        mol_embeddings.append(
            emb.cpu().numpy()
        )

mol_embeddings = np.vstack(mol_embeddings)


print("\nEncoding spectra...\n")

spectra_tensor = torch.tensor(
    np.array(spectra),
    dtype=torch.float
).to(DEVICE)

with torch.no_grad():

    spec_embeddings = spec_encoder(
        spectra_tensor
    )

    spec_embeddings = F.normalize(
        spec_embeddings,
        dim=1
    )

spec_embeddings = spec_embeddings.cpu().numpy()


# ============================================================
# RETRIEVAL
# ============================================================

print("\nComputing similarities...\n")

similarity = spec_embeddings @ mol_embeddings.T

top1 = 0
top5 = 0
top10 = 0

N = len(similarity)

for i in range(N):

    ranking = np.argsort(
        similarity[i]
    )[::-1]

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
