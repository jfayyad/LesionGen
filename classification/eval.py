import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import csv

# Define dataset directory
dataset_dir = '/home/jfayyad/Python_Projects/LesionGen/generated_images/dataset'

# Define label mapping
label_mapping = {
    "Basal cell carcinoma": 1,
    "Melanoma": 4
}

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

# Custom Dataset
class CustomImageDataset(Dataset):
    def __init__(self, dataset_dir, transform, label_mapping):
        self.samples = []
        self.transform = transform
        self.label_mapping = label_mapping

        # Load all samples and their labels
        for label_name, mapped_label in label_mapping.items():
            class_dir = os.path.join(dataset_dir, label_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, mapped_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)

# Create dataset and DataLoader
dataset = CustomImageDataset(dataset_dir=dataset_dir, transform=transform, label_mapping=label_mapping)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 7)  # Model trained with 7 classes
model.load_state_dict(torch.load('/home/jfayyad/Python_Projects/LesionGen/classification/weights/best_model_resnet18.pth'))
model.to(device)
model.eval()

# Evaluate the model and store predictions
correct = 0
total = 0
predictions = []

with torch.no_grad():
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # Store predictions and corresponding labels
        predictions.extend(zip(labels.cpu().numpy(), predicted.cpu().numpy()))

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Accuracy on the dataset: {accuracy:.4f}")

# Display predictions
print("\nPredictions (True Label -> Predicted Label):")
for true_label, pred_label in predictions:
    print(f"{true_label} -> {pred_label}")

