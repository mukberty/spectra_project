import torch
import numpy as np
import faiss
import pickle
from PIL import Image
import torchvision.transforms as T
import pandas as pd

from models.molclip_model import MolCLIP


# -----------------------
# DEVICE
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# LOAD TOKENIZER
# -----------------------
with open("checkpoints/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# -----------------------
# LOAD MODEL
# -----------------------
ckpt = torch.load(
    "checkpoints/molclip_aligned.pt",
    map_location=device
)

model = MolCLIP(
    vocab_size=tokenizer.vocab_size
).to(device)

model.load_state_dict(ckpt, strict=True)
model.eval()


# -----------------------
# LOAD INDEX
# -----------------------
index = faiss.read_index(
    "checkpoints/molclip.index"
)

ids = np.load(
    "checkpoints/molclip_ids.npy",
    allow_pickle=True
)


# -----------------------
# LOAD DATASET
# -----------------------
df = pd.read_parquet(
    "data/processed/curated_subset_named.parquet"
)


# -----------------------
# IMAGE TRANSFORM
# -----------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])


# -----------------------
# QUERY FUNCTION
# -----------------------
def query(ir_path, raman_path, topk=10):

    ir = Image.open(ir_path).convert("RGB")
    raman = Image.open(raman_path).convert("RGB")

    ir = transform(ir).unsqueeze(0).to(device)
    raman = transform(raman).unsqueeze(0).to(device)

    with torch.no_grad():

        emb = model.encode_image(
            ir,
            raman
        )

    emb = emb.cpu().numpy().astype("float32")

    scores, neighbors = index.search(
        emb,
        topk
    )

    print("\nTOP RESULTS\n")

    for rank, idx in enumerate(neighbors[0]):

        smiles = str(ids[idx])

        row = df[
            df["smiles"] == smiles
        ]

        if len(row):

            if "molecule_name" in row.columns:
                name = row.iloc[0]["molecule_name"]
            elif "iupac_name" in row.columns:
                name = row.iloc[0]["iupac_name"]
            else:
                name = smiles

        else:
            name = smiles

        print(
            f"{rank+1:2d}. "
            f"score={scores[0][rank]:.4f}"
        )

        print(name)
        print()


# -----------------------
# EXAMPLE
# -----------------------
query(
    "data/plots/continuous_spectra/molecule_35000_IR.png",
    "data/plots/continuous_spectra/molecule_35000_Raman.png",
    topk=10
)
