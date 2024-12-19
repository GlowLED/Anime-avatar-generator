import torch
from torch import nn



class DiffusionModel(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int=1000) -> None:
        super().__init__()
        self.model = model
        self.timesteps = timesteps
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass