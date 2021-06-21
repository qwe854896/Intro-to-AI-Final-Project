import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset
import os

from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class images_dataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return 124799

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(idx + 1) + '.png')
        try:
            image = Image.open(img_name)
        except Exception as e:
            image = Image.open(os.path.join(self.root_dir, str(1) + '.png'))

        image = image.convert('L')

        if self.transform:
            image = self.transform(image)

        return image

def dataloader(batch_size):
    datapath = "/home/colder66/Documents/CS_subject/intro_AI/Intro-to-AI-Final-Project/dataset/concise"
    transform=transforms.Compose(
        [transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )
    dataset = images_dataset(datapath, transform)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader
