import torch
from test_loader import get_dataloader
import torch.nn as nn
from tqdm import tqdm
import os
from torchvision import models
import numpy as np

img_dir = '/home/jfayyad/Python_Projects/VLMs/diffusers/examples/text_to_image/Future_samples_3'
test_loader = get_dataloader(img_dir=img_dir)

model = models.vit_b_16(pretrained=True)
num_features = model.heads.head.in_features
model.heads.head = nn.Linear(num_features, 7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load weights
model.load_state_dict(torch.load('/home/jfayyad/Python_Projects/LesionGen/classification/weights/best_model_vit.pth', map_location=device))

# Evaluate on the test set
model.eval()
class_correct = np.zeros(7)
class_total = np.zeros(7)

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for label, pred in zip(labels, predicted):
            class_total[label.item()] += 1
            class_correct[label.item()] += (pred == label).item()

# Calculate accuracy
for i in range(7):
    if class_total[i] > 0:
        accuracy = (class_correct[i] / class_total[i]) * 100
        print(f"Class {i} Accuracy: {accuracy:.2f}%")
    else:
        print(f"Class {i} Accuracy: No samples available.")