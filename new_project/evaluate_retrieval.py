import torch
import numpy as np
import faiss
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset

from models.molclip_model import MolCLIP
from utils.image_retrieval_dataset import ImageRetrievalDataset


# -------------------
# setup
# -------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# -------------------
# LOAD MOLCLIP (IMPORTANT FIX)
# -------------------

model = MolCLIP(vocab_size=39).to(device)

ckpt = torch.load(
    "checkpoints/molclip.pt",
    map_location=device
)

model.load_state_dict(ckpt, strict=False)
model.eval()


# -------------------
# LOAD INDEX (MUST MATCH MOLCLIP SPACE)
# -------------------

index = faiss.read_index("checkpoints/molecule.index")
ids = np.load("checkpoints/molecule_ids.npy", allow_pickle=True)

print("Index size:", index.ntotal)


# -------------------
# DATASET (TEST SPLIT ONLY)
# -------------------

dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra"
)

test_idx = np.load("data/splits/test_indices.npy")

test_dataset = Subset(dataset, test_idx)

loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("Test samples:", len(test_dataset))


# -------------------
# EMBEDDING FUNCTION (MOLCLIP SPACE)
# -------------------

def get_embedding(ir_img, raman_img):

    ir_img = ir_img.to(device)
    raman_img = raman_img.to(device)

    with torch.no_grad():
        emb = model.encode_image(ir_img, raman_img)
        emb = F.normalize(emb, dim=1)

    return emb.cpu().numpy().astype("float32")


# -------------------
# EVALUATION
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

        if retrieved_ids[0] == true_id:
            top1 += 1

        if true_id in retrieved_ids:
            top5 += 1

        total += 1


# -------------------
# RESULTS
# -------------------

print("\n===== RETRIEVAL RESULTS =====")
print(f"Top-1 Accuracy: {top1 / total:.4f}")
print(f"Top-5 Accuracy: {top5 / total:.4f}")
print(f"Total samples: {total}")
