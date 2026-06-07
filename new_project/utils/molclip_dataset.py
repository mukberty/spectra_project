import pandas as pd
import torch

from utils.image_retrieval_dataset import ImageRetrievalDataset


class MolCLIPDataset(ImageRetrievalDataset):

    def __init__(self, parquet_path, image_dir):

        super().__init__(
            parquet_path=parquet_path,
            image_dir=image_dir
        )

        self.df = pd.read_parquet(parquet_path)

        smiles_list = self.df["smiles"].astype(str).tolist()

        chars = sorted(
            list(set("".join(smiles_list)))
        )

        self.stoi = {
            c: i + 1
            for i, c in enumerate(chars)
        }

        self.stoi["<pad>"] = 0

        self.itos = {
            v: k
            for k, v in self.stoi.items()
        }

        self.max_len = 256

    def encode_smiles(self, smiles):

        tokens = [
            self.stoi.get(c, 0)
            for c in smiles
        ]

        tokens = tokens[:self.max_len]

        tokens += [0] * (
            self.max_len - len(tokens)
        )

        return torch.tensor(tokens)

    def __getitem__(self, idx):

        ir_img, raman_img, mol_id = \
            super().__getitem__(idx)

        smiles = str(
            self.df.iloc[idx]["smiles"]
        )

        smiles_tokens = self.encode_smiles(
            smiles
        )

        return (
            ir_img,
            raman_img,
            smiles_tokens
        )
