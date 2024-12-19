import torch
import torch.nn as nn
import math
from tools.process import timestep_embedding


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        t_emb = timestep_embedding(x, dim=self.dim, max_period=self.max_period)
        return t_emb


class SEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        
        # Input parameter validation
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")
        if not isinstance(reduction, int) or reduction <= 0 or reduction > in_channels:
            raise ValueError("reduction must be a positive integer and less than or equal to in_channels")
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        v = self.avg_pool(x).view(b, c)
        v = self.fc(v).view(b, c, 1, 1)
        v = self.sigmoid(v)
        return x * v


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout: float=0.2
                 ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      padding_mode="reflect",
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      padding_mode="reflect",
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
            nn.LeakyReLU(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout: float=0.2,
                 t_emb_dim: int=512
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.t_embedding = nn.Sequential(
            nn.Linear(t_emb_dim, in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, in_channels)
        )
        self.convblock = ConvBlock(in_channels, out_channels, dropout)
        self.pooling = nn.MaxPool2d(2, 2)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = x + self.t_embedding(t_emb).reshape(-1, self.in_channels, 1, 1)
        return self.pooling(self.convblock(x))


class DownSampleWithSE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout: float=0.2,
                 t_emb_dim: int=512
                 ):
        super().__init__()
        self.downsample = DownSample(in_channels, out_channels, dropout)
        self.se = SEBlock(out_channels)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.se(self.downsample(x, t_emb))


class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float=0.2, t_emb_dim: int=512) -> None:
        super().__init__()
        self.t_embedding = nn.Sequential(
            nn.Linear(t_emb_dim, in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, in_channels)
        )
        self.convtrans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.convblock = ConvBlock(in_channels, out_channels)
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = x + self.t_embedding(t_emb).reshape(-1, self.in_channels, 1, 1)
        return self.convtrans(self.convblock(x))


class UpSampleWithSE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float=0.2, t_emb_dim: int=512) -> None:
        self.upsample = UpSample(in_channels, out_channels, dropout)
        self.se = SEBlock(out_channels)
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.se(self.upsample(x, t_emb))


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int=64, dropout: float=0.2) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.time_embedding = TimeEmbedding(hidden_channels)

        self.down_layers = []
        self.down_layers.append(DownSampleWithSE(in_channels, hidden_channels))
        self.down_layers.append(DownSampleWithSE(hidden_channels, hidden_channels*2))
        self.down_layers.append(DownSampleWithSE(hidden_channels*2, hidden_channels*4))
        
        self.middle_layer = ConvBlock(hidden_channels*4, hidden_channels*8)

        self.up_layers = []
        self.up_layers.append(UpSampleWithSE(hidden_channels*8, hidden_channels*4))
        self.up_layers.append(UpSampleWithSE(hidden_channels*4, hidden_channels*2))
        self.up_layers.append(UpSampleWithSE(hidden_channels*2, hidden_channels))

        self.out_layer = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        
        t_emb = self.time_embedding(timesteps)

        for layer in self.down_layers:
            x = layer(x, t_emb)

        x = self.middle_layer(x)
        

        for layer in self.up_layers:
            x = layer(x, t_emb)

        return self.out_layer(x)

