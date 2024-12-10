import torch
import torch.nn as nn
from model import UNet

model = UNet()

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
