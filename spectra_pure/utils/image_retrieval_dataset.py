import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as T


class ImageRetrievalDataset(Dataset):

    def __init__(
        self,
        parquet_path,
        image_dir,
        split_ratio=0.2,
        mode="all"
    ):

        self.df = pd.read_parquet(parquet_path)
        self.image_dir = image_dir

        # preserve original row ids
        self.df["global_idx"] = self.df.index

        if mode != "all":

            n = len(self.df)
            split_idx = int(n * (1 - split_ratio))

            if mode == "train":
                self.df = self.df.iloc[:split_idx]

            elif mode == "test":
                self.df = self.df.iloc[split_idx:]

        self.df = self.df.reset_index(drop=True)

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

        print(f"[Dataset] Mode: {mode}")
        print(f"[Dataset] Size: {len(self.df)}")

    def load_img(self, fname):

        path = os.path.join(self.image_dir, fname)

        img = Image.open(path).convert("RGB")

        return self.transform(img)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        global_idx = int(row["global_idx"])

        ir_name = f"molecule_{global_idx}_IR.png"
        raman_name = f"molecule_{global_idx}_Raman.png"

        ir_img = self.load_img(ir_name)
        raman_img = self.load_img(raman_name)

        smiles = str(row["smiles"])

        return ir_img, raman_img, smiles
