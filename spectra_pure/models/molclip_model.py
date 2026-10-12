import torch.nn as nn
import torch.nn.functional as F

from models.image_cnn import ImageCNN
#from models.image_cnn_doublenet import ImageCNN
from models.smiles_encoder import SmilesEncoder


class MolCLIP(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.image_encoder = ImageCNN(
            embed_dim=256
        )

        self.smiles_encoder = SmilesEncoder(
            vocab_size=vocab_size,
            embed_dim=256
        )

    def encode_image(
        self,
        ir_img,
        raman_img
    ):

        z = self.image_encoder(
            ir_img,
            raman_img
        )

        return F.normalize(
            z,
            dim=1
        )

    def encode_smiles(
        self,
        smiles
    ):

        z = self.smiles_encoder(
            smiles
        )

        return F.normalize(
            z,
            dim=1
        )
