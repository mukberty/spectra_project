import io
import random

import numpy as np
import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt

import torchvision.transforms as T

from torch.utils.data import Dataset


class MixtureDataset(Dataset):

    def __init__(
        self,
        parquet_path,
        indices=None,
        image_size=224
    ):

        self.df = pd.read_parquet(
            parquet_path
        )

        if indices is not None:
            self.df = self.df.iloc[indices]

        self.df = self.df.reset_index(drop=True)

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def spectrum_to_image(
        self,
        x_axis,
        spectrum
    ):

        fig = plt.figure(
            figsize=(6, 3)
        )

        plt.plot(
            x_axis,
            spectrum,
            linewidth=1
        )

        plt.gca().invert_xaxis()

        plt.axis("off")

        buf = io.BytesIO()

        plt.savefig(
            buf,
            format="png",
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close(fig)

        buf.seek(0)

        img = Image.open(buf).convert("RGB")

        return self.transform(img)

    def __getitem__(
        self,
        idx
    ):

        row1 = self.df.iloc[idx]

        idx2 = random.randint(
            0,
            len(self.df) - 1
        )

        while idx2 == idx:
            idx2 = random.randint(
                0,
                len(self.df) - 1
            )

        row2 = self.df.iloc[idx2]

        # =====================================================
        # MIXING RATIO
        # =====================================================

        #w = 0.5

        w = random.uniform(
            0.05,
            0.95
        )

        # =====================================================
        # IR
        # =====================================================

        ir1 = np.array(
            row1["continuous_ir"],
            dtype=np.float32
        )

        ir2 = np.array(
            row2["continuous_ir"],
            dtype=np.float32
        )

        ir_mix = (
            w * ir1 +
            (1 - w) * ir2
        )

        # =====================================================
        # RAMAN
        # =====================================================

        ram1 = np.array(
            row1["continuous_raman"],
            dtype=np.float32
        )

        ram2 = np.array(
            row2["continuous_raman"],
            dtype=np.float32
        )

        ram_mix = (
            w * ram1 +
            (1 - w) * ram2
        )

        # =====================================================
        # AXIS
        # =====================================================

        x_axis = np.array(
            row1["x_axis"]
        )

        # =====================================================
        # IMAGES
        # =====================================================

        ir_img = self.spectrum_to_image(
            x_axis,
            ir_mix
        )

        ram_img = self.spectrum_to_image(
            x_axis,
            ram_mix
        )

        # =====================================================
        # RETURN
        # =====================================================

        return (
            ir_img,
            ram_img,
            row1["smiles"],
            row2["smiles"],
            w
        )
