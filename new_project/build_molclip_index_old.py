import torch
import numpy as np
import faiss
import pandas as pd
from torch.utils.data import DataLoader

from models.molclip_model import MolCLIP
from utils.molclip_dataset import MolCLIPDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)


dataset = MolCLIPDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra"
)

loader = DataLoader(dataset, batch_size=64, shuffle=False)


model = MolCLIP(vocab_size=len(dataset.stoi)).to(device)

model.load_state_dict(torch.load("checkpoints/molclip.pt", map_location=device))
model.eval()


df = pd.read_parquet("data/processed/curated_subset_named.parquet")


embeddings = []
ids = []

with torch.no_grad():

    for ir_img, raman_img, smiles in loader:

        ir_img = ir_img.to(device)
        raman_img = raman_img.to(device)

        emb = model.encode_image(ir_img, raman_img)

        embeddings.append(emb.cpu().numpy())

        # use SMILES as identifier
        for s in smiles:
            ids.append(s)


embeddings = np.vstack(embeddings).astype("float32")

faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(256)
index.add(embeddings)


faiss.write_index(index, "checkpoints/molclip.index")

np.save("checkpoints/molclip_ids.npy", np.array(ids))

print("index built")
