from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os


class ImageSet(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.image_path = os.listdir(path)
    
    def __getitem__(self, idx):
        image_name = self.image_path[idx]
        image_item_path = os.path.join(self.image_path, image_name)
        image = Image.open(image_item_path)
        return image
    
    def __len__(self):
        return len(self.image_path)
        
