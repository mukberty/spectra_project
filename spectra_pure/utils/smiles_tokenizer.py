import torch

class SmilesTokenizer:
    """
    Simple character-level tokenizer for SMILES.
    Works well for MolCLIP baseline systems.
    """

    def __init__(self, smiles_list, max_len=128):

        self.max_len = max_len

        # build vocabulary from dataset
        charset = set()
        for s in smiles_list:
            charset.update(list(s))

        charset = sorted(list(charset))

        self.char2idx = {c: i + 1 for i, c in enumerate(charset)}
        self.char2idx["<pad>"] = 0

        self.vocab_size = len(self.char2idx)

        print(f"[Tokenizer] vocab size: {self.vocab_size}")

    def encode(self, smiles):
        """
        Convert single SMILES string → tensor
        """

        ids = []

        for c in smiles:
            if c in self.char2idx:
                ids.append(self.char2idx[c])

        # truncate
        ids = ids[:self.max_len]

        # pad
        ids += [0] * (self.max_len - len(ids))

        return torch.tensor(ids, dtype=torch.long)

    def batch_encode(self, smiles_list):
        return torch.stack([self.encode(s) for s in smiles_list])
