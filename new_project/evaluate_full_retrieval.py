import torch
import faiss
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.image_cnn import ImageCNN
from utils.image_retrieval_dataset import ImageRetrievalDataset


# --------------------------------
# setup
# --------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

dataset = ImageRetrievalDataset(
    parquet_path="data/processed/curated_subset_named.parquet",
    image_dir="data/plots/continuous_spectra"
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False
)

model = ImageCNN(embed_dim=256).to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/image_retrieval.pt",
        map_location=device
    )
)

model.eval()


# --------------------------------
# build embeddings
# --------------------------------

print("Building embeddings...")

embeddings = []
names = []

with torch.no_grad():

    for ir_img, raman_img, mol_name in loader:

        ir_img = ir_img.to(device)
        raman_img = raman_img.to(device)

        emb = model(ir_img, raman_img)
        emb = F.normalize(emb, dim=1)

        embeddings.append(
            emb.cpu().numpy()
        )

        names.extend(list(mol_name))

embeddings = np.vstack(embeddings).astype("float32")

print("Embeddings shape:", embeddings.shape)


# --------------------------------
# build FAISS index
# --------------------------------

index = faiss.IndexFlatIP(256)
index.add(embeddings)

print("Index size:", index.ntotal)


# --------------------------------
# self retrieval evaluation
# --------------------------------

top1 = 0
top5 = 0
total = len(names)

print("Running retrieval evaluation...")

batch_size = 512

for start in range(0, len(embeddings), batch_size):

    end = min(start + batch_size, len(embeddings))

    query = embeddings[start:end]

    scores, idxs = index.search(query, 6)

    for local_i in range(len(query)):

        global_i = start + local_i

        true_name = names[global_i]

        retrieved = []

        for idx in idxs[local_i]:

            if idx == global_i:
                continue

            retrieved.append(names[idx])

            if len(retrieved) == 5:
                break

        if len(retrieved) == 0:
            continue

        if retrieved[0] == true_name:
            top1 += 1

        if true_name in retrieved:
            top5 += 1


print("\n===== FULL DATASET RETRIEVAL =====")
print(f"Top-1 Accuracy: {top1/total:.4f}")
print(f"Top-5 Accuracy: {top5/total:.4f}")
print(f"Total molecules: {total}")


# --------------------------------
# show examples
# --------------------------------

print("\n===== SAMPLE RETRIEVALS =====")

for sample_idx in [0, 100, 1000]:

    scores, idxs = index.search(
        embeddings[sample_idx:sample_idx+1],
        6
    )

    retrieved = []

    for idx in idxs[0]:

        if idx == sample_idx:
            continue

        retrieved.append(names[idx])

        if len(retrieved) == 5:
            break

    print("\nQUERY:")
    print(names[sample_idx])

    print("\nTOP-5:")

    for r in retrieved:
        print("-", r)

    print("-" * 80)
