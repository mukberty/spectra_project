import pickle
import numpy as np
import pandas as pd
import faiss
import torch
import matplotlib.pyplot as plt

from models.molclip_model import MolCLIP


# =====================================================
# SETTINGS
# =====================================================

QUERY_IDX = np.random.randint(39555)
TOP_K = 5

X_MIN = 400
X_MAX = 4000
STEP = 2
GAMMA = 12

x_axis = np.arange(X_MIN, X_MAX + STEP, STEP)


# =====================================================
# LORENTZIAN
# =====================================================

def lorentzian(x, x0, intensity, gamma):
    return intensity * (
        (gamma**2) /
        ((x - x0)**2 + gamma**2)
    )


def generate_spectrum(freqs, intensities):

    y = np.zeros_like(
        x_axis,
        dtype=np.float32
    )

    for f, inten in zip(freqs, intensities):

        if f < X_MIN or f > X_MAX:
            continue

        y += lorentzian(
            x_axis,
            f,
            inten,
            GAMMA
        )

    return y


# =====================================================
# LOAD DATA
# =====================================================

print("Loading parquet...")

df = pd.read_parquet(
    "data/processed/curated_subset_named.parquet"
)

print("Loading FAISS...")

index = faiss.read_index(
    "checkpoints/molclip.index"
)

ids = np.load(
    "checkpoints/molclip_ids.npy",
    allow_pickle=True
)

# =====================================================
# LOAD MODEL
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

with open(
    "checkpoints/tokenizer.pkl",
    "rb"
) as f:
    tokenizer = pickle.load(f)

ckpt = torch.load(
    "checkpoints/molclip_aligned.pt",
    map_location=device
)

model = MolCLIP(
    vocab_size=ckpt["smiles_encoder.token_emb.weight"].shape[0]
).to(device)

model.load_state_dict(
    ckpt,
    strict=False
)

model.eval()

print("Model loaded")


# =====================================================
# BUILD SMILES -> ROW MAP
# =====================================================

smiles_to_row = {}

for idx, row in df.iterrows():

    smiles_to_row[
        str(row["smiles"])
    ] = idx


# =====================================================
# QUERY MOLECULE
# =====================================================

query_smiles = str(
    ids[QUERY_IDX]
)

row_idx = smiles_to_row[
    query_smiles
]

row = df.iloc[row_idx]

freq = np.array(
    row["freq"]
)

raman = np.array(
    row["raman_activity"]
)

query_spec = generate_spectrum(
    freq,
    raman
)

query_spec = query_spec / (
    query_spec.max() + 1e-8
)

# =====================================================
# QUERY EMBEDDING
# =====================================================

query_vec = index.reconstruct(
    QUERY_IDX
).reshape(1, -1)

D, I = index.search(
    query_vec,
    TOP_K + 1
)

# remove self-match
retrieved = I[0][1:]
scores = D[0][1:]


# =====================================================
# PLOT
# =====================================================

# =====================================================
# PLOT
# =====================================================

fig = plt.figure(figsize=(16, 8))

ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
ax2 = plt.subplot2grid((1, 3), (0, 2))

# ----------------------------------------
# spectra
# ----------------------------------------

ax1.plot(
    x_axis,
    query_spec,
    linewidth=3,
    label="Query"
)

retrieval_text = []

query_name = str(row["molecule_name"])

for rank, (idx_faiss, score) in enumerate(
    zip(retrieved, scores),
    start=1
):

    smiles = str(ids[idx_faiss])

    row_idx = smiles_to_row[smiles]

    hit = df.iloc[row_idx]

    spec = generate_spectrum(
        np.array(hit["freq"]),
        np.array(hit["raman_activity"])
    )

    spec = spec / (
        spec.max() + 1e-8
    )

    ax1.plot(
        x_axis,
        spec,
        alpha=0.8,
        label=f"Top-{rank}"
    )

    retrieval_text.append(
        f"{rank}. {hit['molecule_name']}\n"
        f"   sim = {score:.3f}\n"
    )

ax1.set_xlabel(
    "Wavenumber (cm$^{-1}$)"
)

ax1.set_ylabel(
    "Normalized Raman Activity"
)

ax1.set_title(
    "MolCLIP Retrieval Example"
)

ax1.legend()

# ----------------------------------------
# text panel
# ----------------------------------------

ax2.axis("off")

text_block = (
    "QUERY\n"
    "-------------------------\n"
    f"{query_name}\n\n"
    "TOP RETRIEVALS\n"
    "-------------------------\n\n"
    + "\n".join(retrieval_text)
)

ax2.text(
    0,
    1,
    text_block,
    fontsize=10,
    va="top",
    family="monospace"
)

plt.tight_layout()

plt.savefig(
    "retrieval_overlay.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
