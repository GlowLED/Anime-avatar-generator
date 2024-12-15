import torch
import torch.nn as nn


class ConvBlock(nn.module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float=0.2) -> None:
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


class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class UNet(nn.Module):
    def __init__(self, features: int=32, dropout: float=0.2) -> None:

        self.time_embed = nn.Sequential(
            nn.Linear()
        )
        pass
        
        self.ConvBlock1 = ConvBlock(3, features)
        self.Pooling1 = nn.MaxPool2d(2, 2)
        self.ConvBlock2 = ConvBlock(features, features*2, dropout)
        self.Pooling2 = nn.MaxPool2d(2, 2)
        self.ConvBlock3 = ConvBlock(features*2, features*4, dropout)
        self.Pooling3 = nn.MaxPool2d(2, 2)
        self.ConvBlock4 = ConvBlock(features*4, features*8, dropout)
        self.Pooling4 = nn.MaxPool2d(2, 2)
        self.ConvBlock5 = ConvBlock(features*8, features*16, dropout)

        self.UpSample1 = UpSample(features*16, features*8)
        self.ConvBlock6 = ConvBlock(features*16, features*8, dropout)
        self.UpSample2 = UpSample(features*8, features*4)
        self.ConvBlock7 = ConvBlock(features*8, features*4, dropout)
        self.UpSample3 = UpSample(features*4, features*2)
        self.ConvBlock8 = ConvBlock(features*4, features*2, dropout)
        self.UpSample4 = UpSample(features*2, features)
        self.ConvBlock9 = ConvBlock(features*2, features, dropout)
        self.FinalConv = nn.Conv2d(features, 3, 1, 1)
        
    
    def forward(self, input: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:

        state1 = self.ConvBlock1(input)
        state2 = self.ConvBlock2(self.Pooling1(state1))
        state3 = self.ConvBlock3(self.Pooling2(state2))
        state4 = self.ConvBlock4(self.Pooling3(state3))
        output = self.ConvBlock5(self.Pooling4(state4))
        output = self.UpSample1(output)
        output = self.ConvBlock6(torch.cat((state4, output), dim=1))
        output = self.UpSample2(output)
        output = self.ConvBlock7(torch.cat((state3, output), dim=1))
        output = self.UpSample3(output)
        output = self.ConvBlock8(torch.cat((state2, output), dim=1))
        output = self.UpSample4(output)
        output = self.ConvBlock9(torch.cat((state1, output), dim=1))
        output = self.FinalConv(output)

        return output