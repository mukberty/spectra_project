import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from models.molclip_model import MolCLIP
from utils.molclip_dataset import MolCLIPDataset


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using:", device)


dataset = MolCLIPDataset(
    parquet_path=
    "data/processed/curated_subset_named.parquet",

    image_dir=
    "data/plots/continuous_spectra"
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4
)


model = MolCLIP(
    vocab_size=len(dataset.stoi)
).to(device)


# load pretrained image encoder

state = torch.load(
    "checkpoints/image_retrieval.pt",
    map_location=device
)

model.image_encoder.load_state_dict(
    state
)


# freeze image encoder

for p in model.image_encoder.parameters():
    p.requires_grad = False


optimizer = torch.optim.Adam(
    model.smiles_encoder.parameters(),
    lr=1e-4
)


def clip_loss(
    image_emb,
    smiles_emb,
    temp=0.07
):

    logits = (
        image_emb @ smiles_emb.T
    ) / temp

    labels = torch.arange(
        len(image_emb),
        device=device
    )

    loss_i = F.cross_entropy(
        logits,
        labels
    )

    loss_s = F.cross_entropy(
        logits.T,
        labels
    )

    return (
        loss_i + loss_s
    ) / 2


for epoch in range(20):

    model.train()

    total_loss = 0

    for (
        ir_img,
        raman_img,
        smiles
    ) in loader:

        ir_img = ir_img.to(device)
        raman_img = raman_img.to(device)

        smiles = smiles.to(device)

        with torch.no_grad():

            image_emb = (
                model.encode_image(
                    ir_img,
                    raman_img
                )
            )

        smiles_emb = (
            model.encode_smiles(
                smiles
            )
        )

        loss = clip_loss(
            image_emb,
            smiles_emb
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"epoch {epoch+1}: "
        f"{total_loss/len(loader):.4f}"
    )


torch.save(
    model.state_dict(),
    "checkpoints/molclip.pt"
)

print("saved model")
