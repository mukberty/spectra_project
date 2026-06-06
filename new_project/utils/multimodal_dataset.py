import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):

    def __init__(self):

        df = pd.read_parquet(
            "data/processed/continuous_spectra.parquet"
        )

        # convert to proper arrays
        ir = np.stack(df["continuous_ir"].values)
        raman = np.stack(df["continuous_raman"].values)

        # FIX: spectrum is CONCAT, not addition
        spectrum = np.concatenate([ir, raman], axis=1)

        # log transform (safe)
        ir = np.log1p(ir).astype(np.float32)
        raman = np.log1p(raman).astype(np.float32)
        spectrum = np.log1p(spectrum).astype(np.float32)

        self.ir = ir
        self.raman = raman
        self.spec = spectrum

    def __len__(self):
        return len(self.ir)

    def __getitem__(self, idx):

        ir = torch.tensor(self.ir[idx], dtype=torch.float32)
        raman = torch.tensor(self.raman[idx], dtype=torch.float32)
        spec = torch.tensor(self.spec[idx], dtype=torch.float32)

        # CRITICAL FIX: ensure 2D → 3D image format if needed
        # if your CNN expects images, uncomment below:

        # ir = ir.unsqueeze(0).repeat(3, 1)      # fake RGB if needed
        # raman = raman.unsqueeze(0).repeat(3, 1)

        return ir, raman, spec
