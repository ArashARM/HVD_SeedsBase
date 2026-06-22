import torch
import torch.nn as nn

# 
class SeedIdentityEmbedding(nn.Module):
    """
    Learnable identity embedding for each seed/node.

    Idea:
    -----
    Each seed gets its own trainable vector ("ID card").

    Example:
    --------
    If:
        n_seeds = 100
        seed_id_dim = 16

    then PyTorch creates an embedding table of shape:

        (100, 16)

    meaning:
        seed 0 -> 16 learnable numbers
        seed 1 -> 16 learnable numbers
        ...
        seed 99 -> 16 learnable numbers

    These vectors start random and are optimized through backpropagation.

    Purpose:
    --------
    Helps the network learn seed-specific/node-specific behavior.
    """

    def __init__(self, n_seeds, seed_id_dim):
        super().__init__()
        self.seed_id_dim = int(seed_id_dim)
        if self.seed_id_dim > 0:
            self.embedding = nn.Embedding(n_seeds, self.seed_id_dim)
        else:
            self.embedding = None

    def forward(self, n_seeds, device):
        if self.embedding is None:
            return None

        seed_ids = torch.arange(n_seeds, device=device, dtype=torch.long)
        return self.embedding(seed_ids)
