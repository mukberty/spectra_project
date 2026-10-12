# SpectraPure

Deep-learning based molecular retrieval from Raman spectra using a CLIP-style dual encoder.

## Features

- ResNet image encoder
- Transformer SMILES encoder
- Contrastive learning
- FAISS retrieval
- Streamlit application

## Dataset

~39,500 organic molecules with synthetic Raman spectra.

## Pipeline

Spectrum Image
→ Image Encoder

SMILES
→ SMILES Encoder

Shared embedding space

↓

FAISS retrieval
