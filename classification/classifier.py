import torch 
from dataloader import get_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from tqdm import tqdm
import copy
import os
import sys


img_dir = '/home/jfayyad/Python_Projects/VLMs/Datasets/HAM/images'
train_loader, val_loader, test_loader = get_dataloaders('HAM', img_dir= img_dir, batch_size=100)

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 7)  # Update the final layer for 7 classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# def create_sampler(dataset):
#     class_counts = torch.bincount(torch.tensor(dataset.targets))
#     weights = 1. / class_counts.float()
#     sample_weights = weights[torch.tensor(dataset.targets)]
#     sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
#     return sampler


num_epochs = 10
best_val_accuracy = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(num_epochs):
    model.train()
    train_correct = 0
    train_total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = train_correct / train_total

    
    model.eval()
    val_correct = 0
    val_total = 0

    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_accuracy = val_correct / val_total
    
    os.makedirs('weights', exist_ok=True)
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, 'weights/best_model_resnet18.pth')

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

# Load the best model weights
model.load_state_dict(best_model_wts)
