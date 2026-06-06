import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import faiss

from PIL import Image
from torchvision import transforms

from models.image_cnn import ImageCNN


# -------------------
# setup
# -------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

model = ImageCNN(embed_dim=256).to(device)
model.load_state_dict(
    torch.load("checkpoints/image_retrieval.pt", weights_only=True)
)
model.eval()

# FAISS index + metadata
index = faiss.read_index("checkpoints/molecule.index")
ids = np.load("checkpoints/molecule_ids.npy")

df = pd.read_parquet("data/processed/curated_subset_named.parquet")


# -------------------
# image transform
# -------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def load_img(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


# -------------------
# embedding function (FIXED)
# -------------------

def get_embedding(ir_img, raman_img):

    ir_img = ir_img.to(device)
    raman_img = raman_img.to(device)

    with torch.no_grad():
        emb = model(ir_img, raman_img)   # ✅ correct forward pass
        emb = F.normalize(emb, dim=1)

    return emb.cpu().numpy().astype("float32")


# -------------------
# pick sample
# -------------------

idx = 10  # change this anytime

ir_path = f"data/plots/continuous_spectra/molecule_{idx}_IR.png"
raman_path = f"data/plots/continuous_spectra/molecule_{idx}_Raman.png"

ir_img = load_img(ir_path)
raman_img = load_img(raman_path)

query = get_embedding(ir_img, raman_img)


# -------------------
# FAISS search
# -------------------

faiss.normalize_L2(query)

scores, topk = index.search(query, 6)

print("\nTop-5 retrieved molecules:\n")

query_id = idx  # make sure this is defined earlier

printed = 0

for rank, i in enumerate(topk[0]):

    mol_id = ids[i]

    # ❌ remove self-match (VERY IMPORTANT)
    if mol_id == ids[query_id]:
        continue

    row = df[df["molecule_name"] == mol_id]

    if "smiles" in row.columns:
        label = row["smiles"].values[0]
    else:
        label = mol_id

    print(f"{printed+1}. {mol_id} | {label} | score={scores[0][rank]:.4f}")

    printed += 1

    if printed == 5:
        break
