import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


TOP_CLASSES = [
    "5.1.1 Alkanes",
    "5.5 Benzene derivatives",
    "5.6 Heterocycles",
    "5.2.5 Aldehydes/Ketones",
    "5.3.2 Amides",
    "6.5 Alkaloids",
    "5.2.2 Ethers",
    "5.2.1 Alcohols",
    "5.3.1 Amines",
    "5.4.2 Sulfides",
    "5.1.3 Alkenes",
    "5.2.9 Esters",
    "5.4.7 Sulfonyl compounds",
    "5.3.7 Imines",
    "6.2 Peptides",
    "5.2.6 Carboxylic acids",
    "5.1.2 Haloalkanes",
    "6.6 Lipids",
    "5.3.4 Nitro compounds",
    "5.3.13 Nitriles"
]

CLASS_TO_IDX = {
    c: i for i, c in enumerate(TOP_CLASSES)
}

IDX_TO_CLASS = {
    i: c for c, i in CLASS_TO_IDX.items()
}


class SpectraDataset(Dataset):

    def __init__(self):

        df = pd.read_parquet(
            "data/processed/continuous_spectra.parquet"
        )

        ir = np.stack(
            df["continuous_ir"].values
        )

        raman = np.stack(
            df["continuous_raman"].values
        )

        X = np.concatenate(
            [ir, raman],
            axis=1
        )

        X = np.log1p(X)

        Y = np.zeros(
            (len(df), len(TOP_CLASSES)),
            dtype=np.float32
        )

        for i, classes in enumerate(
            df["matched_classes"]
        ):
            for c in classes:

                if c in CLASS_TO_IDX:
                    Y[i, CLASS_TO_IDX[c]] = 1.0

        self.X = X.astype(np.float32)
        self.Y = Y

        pos = self.Y.sum(axis=0)
        neg = len(self.Y) - pos

        self.pos_weight = torch.tensor(
            neg / (pos + 1e-6),
            dtype=torch.float32
        )

        self.label_names = [
            IDX_TO_CLASS[i]
            for i in range(len(TOP_CLASSES))
        ]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.Y[idx])
        )
