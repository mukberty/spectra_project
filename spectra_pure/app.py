import streamlit as st
import torch
import numpy as np
import pandas as pd
import faiss
import pickle

from PIL import Image
import torchvision.transforms as T

import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw

from models.molclip_model import MolCLIP


# =====================================================
# PAGE
# =====================================================

st.set_page_config(
    page_title="MolCLIP Retrieval",
    layout="wide"
)

st.title("MolCLIP Molecular Retrieval")


# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

st.sidebar.write(f"Device: {device}")


# =====================================================
# SPECTRUM SETTINGS
# =====================================================

X_MIN = 400
X_MAX = 4000
STEP = 2
GAMMA = 12

x_axis = np.arange(
    X_MIN,
    X_MAX + STEP,
    STEP
)


# =====================================================
# LORENTZIAN
# =====================================================

def lorentzian(x, x0, intensity, gamma):

    return intensity * (
        (gamma ** 2)
        /
        (
            (x - x0) ** 2
            + gamma ** 2
        )
    )


def generate_spectrum(freqs, intensities):

    y = np.zeros_like(
        x_axis,
        dtype=np.float32
    )

    for f, inten in zip(
        freqs,
        intensities
    ):

        if f < X_MIN or f > X_MAX:
            continue

        y += lorentzian(
            x_axis,
            f,
            inten,
            GAMMA
        )

    y /= (y.max() + 1e-8)

    return y

# =====================================================
# LOAD EVERYTHING
# =====================================================

@st.cache_resource
def load_everything():

    with open(
        "checkpoints/tokenizer.pkl",
        "rb"
    ) as f:

        tokenizer = pickle.load(f)

    checkpoint = torch.load(
        "checkpoints/best_model.pt",
        map_location=device
    )

    model = MolCLIP(
        vocab_size=checkpoint["vocab_size"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    index = faiss.read_index(
        "checkpoints/molclip.index"
    )

    ids = np.load(
        "checkpoints/molclip_ids.npy",
        allow_pickle=True
    )

    lookup = pd.read_parquet(
        "data/processed/curated_subset_named.parquet"
    )

    return (
        tokenizer,
        model,
        index,
        ids,
        lookup,
        checkpoint
    )


(
    tokenizer,
    model,
    index,
    ids,
    lookup,
    checkpoint
) = load_everything()


st.sidebar.success(
    f"Epoch: {checkpoint['epoch']}"
)

st.sidebar.write(
    f"Loss: {checkpoint['loss']:.4f}"
)

st.sidebar.write(
    "Model: checkpoints_backup_old"
)

# =====================================================
# SMILES LOOKUP
# =====================================================

smiles_to_row = {}

for idx, row in lookup.iterrows():

    smiles_to_row[
        str(row["smiles"])
    ] = idx




# =====================================================
# DISPLAY NAME
# =====================================================

def display_name(row):
    if (
        "common_name" in row.index
        and pd.notna(row["common_name"])
        and row["common_name"] != row["smiles"]
    ):
        return row["common_name"]

    if (
        pd.notna(row["molecule_name"])
        and row["molecule_name"] != row["smiles"]
    ):
        return row["molecule_name"]

    return "Unnamed compound"


# =====================================================
# IMAGE TRANSFORM
# =====================================================

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])


# =====================================================
# EMBEDDING
# =====================================================

def embed_image(image):

    x = transform(
        image
    ).unsqueeze(0)

    x = x.to(device)

    dummy_raman = torch.zeros_like(x)

    with torch.no_grad():

        emb = model.encode_image(
            x,
            dummy_raman
        )

    return (
        emb.cpu()
           .numpy()
           .astype("float32")
    )


# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload IR spectrum image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Spectrum",
        width=500
    )

    query_embedding = embed_image(
        image
    )

    scores, indices = index.search(
        query_embedding,
        5
    )

    scores = scores[0]
    indices = indices[0]

    retrieved_smiles = [
        str(ids[i])
        for i in indices
    ]

    names = []

    for smi in retrieved_smiles:

        if smi in smiles_to_row:

            row = lookup.iloc[
                smiles_to_row[smi]
            ]

            names.append(display_name(row))

        else:

            names.append(
                "Unknown"
            )

    # =================================================
    # TOP-5 TABLE
    # =================================================

    st.subheader(
        "Top-5 Retrievals"
    )

    result_df = pd.DataFrame({

        "Rank":
            [1, 2, 3, 4, 5],

        "Molecule":
            names,

        "Similarity":
            scores,

        "SMILES":
            retrieved_smiles
    })

    st.dataframe(
        result_df,
        use_container_width=True
    )

    # =================================================
    # STRUCTURE GRID
    # =================================================

    st.subheader(
        "Top-5 Retrieved Structures"
    )

    cols = st.columns(5)

    for rank in range(5):

        smi = retrieved_smiles[rank]

        mol = Chem.MolFromSmiles(
            smi
        )

        if mol is None:
            continue

        img = Draw.MolToImage(
            mol,
            size=(250, 250)
        )

        with cols[rank]:

            st.image(
                img,
                use_container_width=True
            )

            st.markdown(
                f"**Top-{rank+1}**"
            )

            st.caption(
                names[rank]
            )

            st.caption(
                f"Similarity = {scores[rank]:.3f}"
            )

    # =================================================
    # OVERLAY
    # =================================================

     # =================================================
    # OVERLAY
    # =================================================

    st.subheader(
        "Query vs Retrieved Spectra"
    )

    plot_options = [
        "Uploaded Spectrum"
    ] + [
        f"Top-{i+1}: {names[i]}"
        for i in range(5)
    ]

    selected_plots = st.multiselect(
        "Select spectra to display",
        options=plot_options,
        default=plot_options
    )

    fig_overlay, ax_overlay = plt.subplots(
        figsize=(12, 6)
ls -lt --time=birth
    )

    # ---------------------------------
    # Uploaded spectrum (black curve)
    # ---------------------------------

    query_np = np.array(image)

    if len(query_np.shape) == 3:
        query_np = query_np.mean(axis=2)

    query_np = query_np.astype(np.float32)

    query_np -= query_np.min()

    if query_np.max() > 0:
        query_np /= query_np.max()

    query_x = np.linspace(
        X_MIN,
        X_MAX,
        query_np.shape[1]
    )

    query_y = query_np.mean(axis=0)

    # invert because dark peaks usually
    # correspond to higher intensity

    query_y = 1.0 - query_y

    query_y -= query_y.min()

    if query_y.max() > 0:
        query_y /= query_y.max()

    if "Uploaded Spectrum" in selected_plots:

        ax_overlay.plot(
            query_x,
            query_y,
            color="#0B5CAD",      # deep blue
            linewidth=2.2,
            alpha=0.95,
            solid_capstyle="round",
            antialiased=True,
            zorder=100,
            label="Uploaded Spectrum"
        )

    # ---------------------------------
    # Retrieved spectra
    # ---------------------------------

    for rank in range(5):

        label = plot_options[rank + 1]

        if label not in selected_plots:
            continue

        smi = retrieved_smiles[rank]

        row_overlay = lookup.iloc[
            smiles_to_row[smi]
        ]

        spec = generate_spectrum(
            np.array(
                row_overlay["freq"]
            ),
            np.array(
                row_overlay["raman_activity"]
            )
        )

        ax_overlay.plot(
            x_axis,
            spec,
            linewidth=2,
            alpha=0.8,
            label=f"{label} ({scores[rank]:.3f})"
        )

    ax_overlay.set_xlabel(
        "Wavenumber (cm⁻¹)"
    )

    ax_overlay.set_ylabel(
        "Normalized Intensity"
    )

    ax_overlay.set_title(
        "Uploaded Spectrum vs Retrieved Molecules"
    )

    ax_overlay.legend()

    st.pyplot(
        fig_overlay
    )

    # =================================================
    # SELECT MOLECULE
    # =================================================

    selected_rank = st.selectbox(
        "Inspect Molecule",
        [1, 2, 3, 4, 5]
    )

    chosen_idx = selected_rank - 1

    chosen_smiles = retrieved_smiles[
        chosen_idx
    ]

    chosen_score = scores[
        chosen_idx
    ]

    row = lookup.iloc[
        smiles_to_row[
            chosen_smiles
        ]
    ]

    title = display_name(row)

    st.header(title)

    st.write(
        f"Similarity score: {chosen_score:.4f}"
    )

    st.code(
        chosen_smiles
    )

    # =================================================
    # LARGE STRUCTURE
    # =================================================

    mol = Chem.MolFromSmiles(
        chosen_smiles
    )

    if mol is not None:

        mol_img = Draw.MolToImage(
            mol,
            size=(700, 350)
        )

        st.image(
            mol_img,
            caption=title
        )

    # =================================================
    # IR
    # =================================================

    st.subheader(
        "IR Spectrum"
    )

    ir_spec = generate_spectrum(
        np.array(
            row["freq"]
        ),
        np.array(
            row["ir_intensity"]
        )
    )

    fig_ir, ax_ir = plt.subplots(
        figsize=(10, 4)
    )

    ax_ir.plot(
        x_axis,
        ir_spec,
        linewidth=2
    )

    ax_ir.set_xlabel(
        "Wavenumber (cm⁻¹)"
    )

    ax_ir.set_ylabel(
        "Intensity"
    )

    ax_ir.set_title(title)

    st.pyplot(
        fig_ir
    )

    # =================================================
    # RAMAN
    # =================================================

    st.subheader(
        "Raman Spectrum"
    )

    raman_spec = generate_spectrum(
        np.array(
            row["freq"]
        ),
        np.array(
            row["raman_activity"]
        )
    )

    fig_ram, ax_ram = plt.subplots(
        figsize=(10, 4)
    )

    ax_ram.plot(
        x_axis,
        raman_spec,
        linewidth=2
    )

    ax_ram.set_xlabel(
        "Wavenumber (cm⁻¹)"
    )

    ax_ram.set_ylabel(
        "Activity"
    )

    ax_ram.set_title(title)

    st.pyplot(
        fig_ram
    )
