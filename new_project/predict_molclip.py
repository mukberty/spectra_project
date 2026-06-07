import torch
import numpy as np
import faiss
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.molclip_model import MolCLIP


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)


dataset = pd.read_parquet("data/processed/curated_subset_named.parquet")


model = MolCLIP(vocab_size=200).to(device)

model.load_state_dict(torch.load("checkpoints/molclip.pt", map_location=device))
model.eval()


index = faiss.read_index("checkpoints/molclip.index")
ids = np.load("checkpoints/molclip_ids.npy", allow_pickle=True)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def get_embedding(ir_path, raman_path):

    ir = Image.open(ir_path).convert("RGB")
    raman = Image.open(raman_path).convert("RGB")

    ir = transform(ir).unsqueeze(0).to(device)
    raman = transform(raman).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(ir, raman)

    emb = F.normalize(emb, dim=1)

    return emb.cpu().numpy().astype("float32")


IR_IMG = "example_ir.png"
RAMAN_IMG = "example_raman.png"

query = get_embedding(IR_IMG, RAMAN_IMG)

faiss.normalize_L2(query)

scores, topk = index.search(query, 10)


print("\nTop-10 molecules:\n")

for i, idx in enumerate(topk[0]):

    smiles = ids[idx]

    row = dataset[dataset["smiles"] == smiles]

    if len(row):
        name = row["molecule_name"].values[0]
    else:
        name = "Unknown"

    print(f"{i+1}. {name} | {smiles} | score={scores[0][i]:.4f}")
