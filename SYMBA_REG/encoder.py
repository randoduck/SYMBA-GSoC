import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sinekan import SineKAN


class SineKANEncoder(nn.Module):
    def __init__(self, in_dim: int = 10, embed_dim: int = 96,
                 grid_size: int = 5, device: str = 'cpu'):
        super().__init__()
        self.point_net = SineKAN(
            [in_dim, 64, 128, 256], grid_size=grid_size, device=device
        )
        self.final = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        h = self.point_net(x.reshape(B * N, D)).reshape(B, N, 256)
        return self.final(h.max(dim=1).values)
