import pickle
import faiss
import numpy as np
import torch

from models.molclip_model import MolCLIP
from utils.image_retrieval_dataset import ImageRetrievalDataset

CHECKPOINT = "checkpoints/best_model.pt"
INDEX_PATH = "checkpoints/molclip.index"
IDS_PATH = "checkpoints/molclip_ids.npy"

PARQUET_PATH = "data/processed/curated_subset_named.parquet"
IMAGE_DIR = "data/plots/continuous_spectra"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load checkpoint
# -------------------------

ckpt = torch.load(
    CHECKPOINT,
    map_location=device
)

model = MolCLIP(
    vocab_size=ckpt["vocab_size"]
).to(device)

model.load_state_dict(
    ckpt["model_state_dict"]
)

model.eval()

# -------------------------
# Load tokenizer
# -------------------------

with open("checkpoints/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -------------------------
# Load dataset
# -------------------------

dataset = ImageRetrievalDataset(
    parquet_path=PARQUET_PATH,
    image_dir=IMAGE_DIR,
    mode="all"
)

# -------------------------
# Load index
# -------------------------

index = faiss.read_index(INDEX_PATH)
ids = np.load(
    IDS_PATH,
    allow_pickle=True
)

# -------------------------
# Test several molecules
# -------------------------

test_indices = [0, 1, 2, 100, 1000]

for dataset_idx in test_indices:

    ir_img, raman_img, true_smiles = dataset[dataset_idx]

    ir_img = ir_img.unsqueeze(0).to(device)
    raman_img = raman_img.unsqueeze(0).to(device)

    with torch.no_grad():

        z_img = model.encode_image(
            ir_img,
            raman_img
        )

        z_img = z_img.cpu().numpy().astype("float32")

    scores, retrieved = index.search(
        z_img,
        10
    )

    print("\n" + "=" * 70)
    print("DATASET INDEX:", dataset_idx)
    print("TRUE SMILES :", true_smiles)

    print("\nTOP 10:")

    for rank in range(10):

        idx = retrieved[0][rank]

        print(
            f"{rank+1:2d}. "
            f"score={scores[0][rank]:.4f} "
            f"{ids[idx]}"
        )

    print(
        "\nTRUE IN TOP 10:",
        str(true_smiles) in [
            str(x) for x in ids[retrieved[0]]
        ]
    )
