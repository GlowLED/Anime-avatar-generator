import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from diffusion.model import UNet
from diffusion.diffusion import DiffusionModel
from tools.process import sample_x_t
from diffusion.data import ImageSet
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


loss = nn.MSELoss()
model = DiffusionModel(UNet(3, 3)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  
dataset = ImageSet("./dataset/256*256", transform=transforms.ToTensor())



def train(diffusion_model: DiffusionModel, dataset, optimizer, loss, epochs=10):
    diffusion_model.train()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  
    for epoch in range(epochs):
        for images in tqdm(dataloader, total=len(dataloader)):
            timesteps = torch.randint(0, diffusion_model.t, (images.shape[0],)).to(device)
            x_0 = images.to(device)
            x_t = sample_x_t(x_0, timesteps, alpha=1.0)
            optimizer.zero_grad()
            x_pred = diffusion_model(x_t, timesteps)
            loss_val = loss(x_pred, x_0)
            loss_val.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss_val.item()}")
    diffusion_model.eval()




