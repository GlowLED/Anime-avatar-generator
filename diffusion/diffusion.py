import torch
from torch import nn
from tools.process import sample_noise



class DiffusionModel(nn.Module):
    def __init__(self, model: nn.Module, T: int=1000) -> None:
        super().__init__()
        self.model = model
        self.T = T
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x_t, t)
    