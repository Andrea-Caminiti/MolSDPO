import torch 
import torch.nn as nn

class PairwiseRBF(nn.Module):
    """RBF on pairwise distances, returns (B, N, N, num_rbf)."""
    def __init__(self, num_rbf=16, cutoff=10.0, device=None):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        # centers + widths
        centers = torch.linspace(0.0, cutoff, num_rbf)
        widths = (centers[1] - centers[0]) * 1.0
        self.register_buffer("centers", centers)
        self.register_buffer("widths", torch.ones_like(centers) * widths)

    def forward(self, coords, batched=False):
        # coords: (B, N, 3)
        if batched:
            d = coords.unsqueeze(3) - coords.unsqueeze(2)
        else: 
            d = coords.unsqueeze(2) - coords.unsqueeze(1)
              # (B,N,N,3)
        dist = torch.sqrt((d**2).sum(-1) + 1e-8)  # (B,N,N)
        # radial basis
        diff = dist[..., None] - self.centers  # (B,N,N,num_rbf)
        rbf = torch.exp(-0.5 * (diff / (self.widths + 1e-8))**2)
        return rbf  # (B,N,N,num_rbf)