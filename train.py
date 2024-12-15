import torch
import torch.nn as nn
from model import UNet
from process import add_noise
from tqdm import tqdm

model = UNet()

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())


def train(model, dataloader, loss_fn, optimizer, epoch=1, T=512):
    for epc in range(epoch):
        for images in dataloader:
            for t in range(T):
                generated_images = model(images)
                

