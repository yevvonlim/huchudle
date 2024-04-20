import torch.nn as nn
from diffusers.models.embeddings import TextTimeEmbedding
import torch 
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F



class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(self.emb(timestep)))
        if emb.dim() == 1:
            scale, shift = torch.chunk(emb, 2)
        else:
            scale, shift = torch.chunk(emb, 2, dim=1)
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        x = self.norm(x) * (1 + scale) + shift
        return x

