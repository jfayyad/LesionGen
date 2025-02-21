import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to get dataloader
def get_dataloader(batch_size=32, img_dir=None, transform=transform, return_dataset=False):
    if not img_dir or not os.path.exists(img_dir):
        raise ValueError(f"Provided image directory '{img_dir}' is invalid or does not exist.")

    dataset = datasets.ImageFolder(root=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if return_dataset:
        return dataset
    else:
        return dataloader