import re
import numpy as np
import pandas as pd

import torch

from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset

from utils.dataset import (
    TOP_CLASSES,
    CLASS_TO_IDX
)

class SpectrumImageDataset(Dataset):

    def __init__(self):

        self.df = pd.read_parquet(
            "data/processed/continuous_spectra.parquet"
        )

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        ir_path = (
            f"data/plots/continuous_spectra/"
            f"molecule_{idx}_IR.png"
        )

        raman_path = (
            f"data/plots/continuous_spectra/"
            f"molecule_{idx}_Raman.png"
        )

        ir = Image.open(ir_path).convert("L")
        raman = Image.open(raman_path).convert("L")

        ir = self.transform(ir)
        raman = self.transform(raman)

        x = torch.cat(
            [
                ir,
                raman,
                torch.zeros_like(ir)
            ],
            dim=0
        )

        y = np.zeros(
            len(TOP_CLASSES),
            dtype=np.float32
        )

        classes = self.df.iloc[idx][
            "matched_classes"
        ]

        for c in classes:

            if c in CLASS_TO_IDX:
                y[
                    CLASS_TO_IDX[c]
                ] = 1

        return x, torch.tensor(y)
