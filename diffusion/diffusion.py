import torch
from torch import nn
from tools.process import sample_noise



class DiffusionModel(nn.Module):
    def __init__(self, model: nn.Module, t: int=1000) -> None:
        super().__init__()
        self.model = model
        self.t = t
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x_t, t)
    
    @torch.no_grad()
    def generate(self, shape: tuple, iter: int=100) -> torch.Tensor:
        x = sample_noise(shape)
        iters = torch.tensor([iter]*shape[0], dtype=torch.float32)
        for i in range(iter):
            x -= self.model(x, iters-i)
        return x

    
    