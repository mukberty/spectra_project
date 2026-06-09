import torch
import faiss
import numpy as np
import torch.nn.functional as F

from models.image_cnn import ImageCNN
from utils.image_retrieval_dataset import ImageRetrievalDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ImageCNN(embed_dim=256).to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/image_retrieval.pt",
        map_location=device
    )
)

model.eval()

index = faiss.read_index(
    "checkpoints/full_molecule.index"
)

names = np.load(
    "checkpoints/full_molecule_names.npy",
    allow_pickle=True
)

dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra"
)

query_idx = int(input("Query index: "))

ir_img, raman_img, true_name = dataset[query_idx]

ir_img = ir_img.unsqueeze(0).to(device)
raman_img = raman_img.unsqueeze(0).to(device)

with torch.no_grad():

    emb = model(ir_img, raman_img)
    emb = F.normalize(emb, dim=1)

query = emb.cpu().numpy().astype("float32")

scores, idxs = index.search(query, 5)

print("\nTRUE MOLECULE:")
print(true_name)

print("\nTOP-5 CANDIDATES:\n")

for rank, (idx, score) in enumerate(
    zip(idxs[0], scores[0]),
    start=1
):

    print(f"{rank}. {names[idx]}")
    print(f"   similarity = {score:.4f}\n")
