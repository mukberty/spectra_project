import torch
import numpy as np
import faiss
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.image_cnn import ImageCNN
from utils.image_retrieval_dataset import ImageRetrievalDataset


# -------------------
# setup
# -------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

model = ImageCNN(embed_dim=256).to(device)
model.load_state_dict(
    torch.load(
        "checkpoints/image_retrieval.pt",
        map_location=device,
        weights_only=True
    )
)
model.eval()

index = faiss.read_index("checkpoints/molecule.index")
ids = np.load("checkpoints/molecule_ids.npy", allow_pickle=True)

dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra"
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
)


# -------------------
# embedding helper
# -------------------

def get_embedding(ir_img, raman_img):

    ir_img = ir_img.to(device)
    raman_img = raman_img.to(device)

    with torch.no_grad():

        emb = model(ir_img, raman_img)
        emb = F.normalize(emb, dim=1)

    return emb.cpu().numpy().astype("float32")


# -------------------
# evaluation
# -------------------

top1 = 0
top5 = 0
total = 0

for ir_img, raman_img, mol_id in loader:

    query = get_embedding(ir_img, raman_img)

    faiss.normalize_L2(query)

    scores, topk = index.search(query, 5)

    for i in range(len(mol_id)):

        true_id = mol_id[i]

        retrieved_ids = [ids[j] for j in topk[i]]

        # top-1
        if retrieved_ids[0] == true_id:
            top1 += 1

        # top-5
        if true_id in retrieved_ids[:5]:
            top5 += 1

        total += 1


# -------------------
# results
# -------------------

print("\n===== RETRIEVAL RESULTS =====")
print(f"Top-1 Accuracy: {top1 / total:.4f}")
print(f"Top-5 Accuracy: {top5 / total:.4f}")
print(f"Total samples: {total}")
