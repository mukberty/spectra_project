import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as T


class ImageRetrievalDataset(Dataset):

    def __init__(self, parquet_path, image_dir):

        self.df = pd.read_parquet(parquet_path)
        self.image_dir = image_dir

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

        # molecule id = row index OR explicit column
        if "molecule_name" in self.df.columns:
            self.ids = self.df["molecule_name"].values
        else:
            self.ids = self.df.index.values

    def load_img(self, fname):
        path = os.path.join(self.image_dir, fname)
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        mol_id = self.ids[idx]

        ir_name = f"molecule_{idx}_IR.png"
        raman_name = f"molecule_{idx}_Raman.png"

        ir_img = self.load_img(ir_name)
        raman_img = self.load_img(raman_name)

        return ir_img, raman_img, mol_id
