import os
import tempfile
import pickle
import numpy as np
import pandas as pd
import faiss
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from rdkit import Chem
from rdkit.Chem import Draw

from models.molclip_model import MolCLIP


# =====================================================
# SETTINGS
# =====================================================

QUERY_IDX = np.random.randint(39555)          # change this to try different molecules
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
        (gamma ** 2)
        / ((x - x0) ** 2 + gamma ** 2)
    )


def generate_spectrum(freqs, intensities):

    spectrum = np.zeros_like(
        x_axis,
        dtype=np.float32
    )

    for f, inten in zip(freqs, intensities):

        if f < X_MIN or f > X_MAX:
            continue

        spectrum += lorentzian(
            x_axis,
            f,
            inten,
            GAMMA
        )

    return spectrum


# =====================================================
# RDKit DRAWING
# =====================================================

def save_structure(smiles, filename):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return False

    img = Draw.MolToImage(
        mol,
        size=(500, 350)
    )

    img.save(filename)

    return True


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
# BUILD SMILES MAP
# =====================================================

print("Building lookup table...")

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

query_row_idx = smiles_to_row[
    query_smiles
]

query_row = df.iloc[
    query_row_idx
]

query_name = str(
    query_row["molecule_name"]
)

print("\nQUERY:")
print(query_name)

query_spec = generate_spectrum(
    np.array(query_row["freq"]),
    np.array(query_row["raman_activity"])
)

query_spec = query_spec / (
    query_spec.max() + 1e-8
)

# =====================================================
# RETRIEVE NEIGHBORS
# =====================================================

query_vec = index.reconstruct(
    QUERY_IDX
).reshape(1, -1)

D, I = index.search(
    query_vec,
    TOP_K + 1
)

retrieved = I[0][1:]
scores = D[0][1:]

# =====================================================
# TEMP DIRECTORY FOR STRUCTURES
# =====================================================

tmpdir = tempfile.mkdtemp()

query_img = os.path.join(
    tmpdir,
    "query.png"
)

save_structure(
    query_smiles,
    query_img
)

# =====================================================
# FIGURE
# =====================================================

fig = plt.figure(
    figsize=(22, 12)
)

# =====================================================
# QUERY STRUCTURE
# =====================================================

ax = plt.subplot2grid(
    (2, 6),
    (0, 0)
)

ax.imshow(
    mpimg.imread(query_img)
)

ax.axis("off")

ax.set_title(
    f"QUERY\n{query_name[:40]}",
    fontsize=10
)

# =====================================================
# SPECTRUM AXIS
# =====================================================

ax_spec = plt.subplot2grid(
    (2, 6),
    (1, 0),
    colspan=6
)

ax_spec.plot(
    x_axis,
    query_spec,
    linewidth=3,
    label="QUERY",
    color="black"
)

# =====================================================
# TOP RETRIEVALS
# =====================================================

for rank, (idx_faiss, score) in enumerate(
    zip(retrieved, scores),
    start=1
):

    smiles = str(
        ids[idx_faiss]
    )

    row_idx = smiles_to_row[
        smiles
    ]

    hit = df.iloc[row_idx]

    hit_name = str(
        hit["molecule_name"]
    )

    # -----------------------------
    # structure image
    # -----------------------------

    img_file = os.path.join(
        tmpdir,
        f"hit_{rank}.png"
    )

    save_structure(
        smiles,
        img_file
    )

    ax_struct = plt.subplot2grid(
        (2, 6),
        (0, rank)
    )

    ax_struct.imshow(
        mpimg.imread(img_file)
    )

    ax_struct.axis("off")

    ax_struct.set_title(
        f"TOP-{rank}\n{hit_name[:40]}\n(sim={score:.3f})",
        fontsize=8
    )

    # -----------------------------
    # spectrum
    # -----------------------------

    spec = generate_spectrum(
        np.array(hit["freq"]),
        np.array(hit["raman_activity"])
    )

    spec = spec / (
        spec.max() + 1e-8
    )

    ax_spec.plot(
        x_axis,
        spec,
        alpha=0.8,
        label=f"Top-{rank}"
    )

# =====================================================
# FINALIZE PLOT
# =====================================================

ax_spec.set_xlabel(
    "Wavenumber (cm$^{-1}$)"
)

ax_spec.set_ylabel(
    "Normalized Raman Activity"
)

ax_spec.set_title(
    "MolCLIP Retrieval: Query vs Top-5 Retrieved Molecules"
)

ax_spec.legend()

plt.tight_layout()

outfile = f"{QUERY_IDX}_retrieval.png"

plt.savefig(
    outfile,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nSaved {outfile}")

print(
    "\nSaved retrieval_overlay.png"
)

plt.show()
