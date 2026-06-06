import torch
import numpy as np
import faiss
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.image_cnn import ImageCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# LOAD MODEL
# -----------------------

model = ImageCNN(embed_dim=256).to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/image_retrieval.pt",
        map_location=device,
        weights_only=True
    )
)

model.eval()


# -----------------------
# LOAD INDEX
# -----------------------

index = faiss.read_index("checkpoints/molecule.index")

ids = np.load(
    "checkpoints/molecule_ids.npy",
    allow_pickle=True
)

df = pd.read_parquet(
    "data/processed/curated_subset_named.parquet"
)


# -----------------------
# IMAGE PREPROCESSING
# -----------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# -----------------------
# EMBEDDING FUNCTION
# -----------------------

def get_embedding(ir_path, raman_path):

    ir_img = Image.open(ir_path).convert("RGB")
    raman_img = Image.open(raman_path).convert("RGB")

    ir_img = transform(ir_img).unsqueeze(0).to(device)
    raman_img = transform(raman_img).unsqueeze(0).to(device)

    with torch.no_grad():

        emb = model(ir_img, raman_img)

        emb = F.normalize(emb, dim=1)

    return emb.cpu().numpy().astype("float32")


# -----------------------
# QUERY
# -----------------------

IR_IMAGE = "example_ir.png"
RAMAN_IMAGE = "example_raman.png"

query = get_embedding(IR_IMAGE, RAMAN_IMAGE)

faiss.normalize_L2(query)

scores, topk = index.search(query, 10)


print("\nTop-10 Candidate Molecules\n")

for rank, idx in enumerate(topk[0]):

    mol_name = ids[idx]

    row = df[df["molecule_name"] == mol_name]

    if len(row):

        iupac = row["molecule_name"].values[0]

        if "smiles" in row.columns:
            smiles = row["smiles"].values[0]
        else:
            smiles = "N/A"

    else:

        iupac = mol_name
        smiles = "N/A"

    print(
        f"{rank+1:2d}. "
        f"score={scores[0][rank]:.4f}\n"
        f"    IUPAC: {iupac}\n"
        f"    SMILES: {smiles}\n"
    )
