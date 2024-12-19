import torch
import math

@torch.no_grad()
def timestep_embedding(timesteps, dim, max_period=10000):

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)

    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


@torch.no_grad()
def sample_x_t(x_0: torch.Tensor, timesteps: torch.Tensor, alpha: float) -> tuple:
    noise = torch.randn_like(x_0)
    pass


@torch.no_grad()
def sample_noise(shape: tuple) -> torch.Tensor:
    return torch.randn(shape)
