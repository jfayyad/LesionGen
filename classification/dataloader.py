import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, annotations, img_dir, transform=None, label_mapping=None, file_extension=".jpg"):
        self.annotations = annotations
        self.img_dir = img_dir
        self.transform = transform
        self.label_mapping = label_mapping
        self.file_extension = file_extension

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index]['image_id'] + self.file_extension)
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image {img_path} not found.")
            return None, None

        label = self.annotations.iloc[index]['dx']
        label = self.label_mapping[label]
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        return image, label

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Updated get_dataloaders function with target class filtering and oversampling
def get_dataloaders(dataset_name, batch_size=32, img_dir=None, transform=transform, return_dataset=False, target_classes=None, augment_underrepresented=True):
    label_mapping = {
        'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6
    }

    def load_and_balance_data(csv_file, target_classes, augment_underrepresented):
        # Load dataset
        annotations = pd.read_csv(csv_file)

        # Filter for target classes
        if target_classes:
            annotations = annotations[annotations['dx'].isin(target_classes)]

        # Calculate augmentation rate for each class based on filtered data
        if augment_underrepresented:
            class_counts = annotations['dx'].value_counts()
            max_count = class_counts.max()
            data_aug_rate = {label: (max_count // count) for label, count in class_counts.items()}
            print("Data Augmentation Rates:", data_aug_rate)

            # Apply the calculated augmentation rates
            augmented_data = []
            for label, rate in data_aug_rate.items():
                class_samples = annotations[annotations['dx'] == label]
                if rate > 1:
                    augmented_data.append(pd.concat([class_samples] * rate, ignore_index=True))
                else:
                    augmented_data.append(class_samples)
            annotations = pd.concat(augmented_data, ignore_index=True)
        
        return annotations

    if dataset_name == 'HAM':
        train_csv_file = '/home/jfayyad/Python_Projects/VLMs/Datasets/HAM/splits/train.csv'
        val_csv_file = '/home/jfayyad/Python_Projects/VLMs/Datasets/HAM/splits/val.csv'
        test_csv_file = '/home/jfayyad/Python_Projects/VLMs/Datasets/HAM/splits/test.csv'

        # Load, filter, and balance training data
        train_annotations = load_and_balance_data(train_csv_file, target_classes, augment_underrepresented= False)
        val_annotations = pd.read_csv(val_csv_file)
        if target_classes is not None:
            val_annotations = val_annotations[val_annotations['dx'].isin(target_classes)]
        test_annotations = pd.read_csv(test_csv_file)
        if target_classes is not None:
            test_annotations = test_annotations[test_annotations['dx'].isin(target_classes)]

    elif dataset_name == 'DMF':
        train_csv_file = 'Datasets/DMF/splits/train.csv'
        val_csv_file = 'Datasets/DMF/splits/val.csv'
        test_csv_file = 'Datasets/DMF/splits/test.csv'

        train_annotations = load_and_balance_data(train_csv_file, target_classes, augment_underrepresented)
        val_annotations = pd.read_csv(val_csv_file)
        val_annotations = val_annotations[val_annotations['dx'].isin(target_classes)]
        test_annotations = pd.read_csv(test_csv_file)
        test_annotations = test_annotations[test_annotations['dx'].isin(target_classes)]

    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    # Create datasets
    train_dataset = CustomDataset(annotations=train_annotations, img_dir=img_dir, transform=transform, label_mapping=label_mapping, file_extension=".jpg" if dataset_name == 'HAM' else ".png")
    val_dataset = CustomDataset(annotations=val_annotations, img_dir=img_dir, transform=transform, label_mapping=label_mapping, file_extension=".jpg" if dataset_name == 'HAM' else ".png")
    test_dataset = CustomDataset(annotations=test_annotations, img_dir=img_dir, transform=transform, label_mapping=label_mapping, file_extension=".jpg" if dataset_name == 'HAM' else ".png")

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    if return_dataset:
        return train_dataset, val_dataset, test_dataset
    else:
        return train_loader, val_loader, test_loader


# import os
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from torchvision import transforms
# from PIL import Image

# class CustomDataset(Dataset):
#     def __init__(self, annotations, img_dir, transform=None, label_mapping=None, file_extension=".jpg"):
#         self.annotations = annotations
#         self.img_dir = img_dir
#         self.transform = transform
#         self.label_mapping = label_mapping
#         self.file_extension = file_extension

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, index):
#         img_path = os.path.join(self.img_dir, self.annotations.iloc[index]['image_id'] + self.file_extension)
#         try:
#             image = Image.open(img_path).convert("RGB")
#         except FileNotFoundError:
#             print(f"Warning: Image {img_path} not found.")
#             return None, None

#         label = self.annotations.iloc[index]['dx']
#         label = self.label_mapping[label]
#         label = torch.tensor(label, dtype=torch.long)

#         if self.transform:
#             image = self.transform(image)

#         return image, label

# # Define image transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# # Updated get_dataloaders function with target class filtering and WeightedRandomSampler
# def get_dataloaders(dataset_name, batch_size=32, img_dir=None, transform=transform, return_dataset=False, target_classes=None):
#     label_mapping = {
#         'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6
#     }

#     def load_data(csv_file, target_classes):
#         # Load dataset
#         annotations = pd.read_csv(csv_file)

#         # Filter for target classes
#         if target_classes:
#             annotations = annotations[annotations['dx'].isin(target_classes)]

#         return annotations

#     if dataset_name == 'HAM':
#         train_csv_file = 'Datasets/HAM/splits/train.csv'
#         val_csv_file = 'Datasets/HAM/splits/val.csv'
#         test_csv_file = 'Datasets/HAM/splits/test.csv'

#         train_annotations = load_data(train_csv_file, target_classes)
#         val_annotations = load_data(val_csv_file, target_classes)
#         test_annotations = load_data(test_csv_file, target_classes)

#     elif dataset_name == 'DMF':
#         train_csv_file = 'Datasets/DMF/splits/train.csv'
#         val_csv_file = 'Datasets/DMF/splits/val.csv'
#         test_csv_file = 'Datasets/DMF/splits/test.csv'

#         train_annotations = load_data(train_csv_file, target_classes)
#         val_annotations = load_data(val_csv_file, target_classes)
#         test_annotations = load_data(test_csv_file, target_classes)

#     else:
#         raise ValueError(f"Dataset '{dataset_name}' not supported.")

#     # Create datasets
#     train_dataset = CustomDataset(annotations=train_annotations, img_dir=img_dir, transform=transform, label_mapping=label_mapping, file_extension=".jpg" if dataset_name == 'HAM' else ".png")
#     val_dataset = CustomDataset(annotations=val_annotations, img_dir=img_dir, transform=transform, label_mapping=label_mapping, file_extension=".jpg" if dataset_name == 'HAM' else ".png")
#     test_dataset = CustomDataset(annotations=test_annotations, img_dir=img_dir, transform=transform, label_mapping=label_mapping, file_extension=".jpg" if dataset_name == 'HAM' else ".png")

#     # Calculate class weights for the sampler
#     class_counts = train_annotations['dx'].value_counts()
#     class_weights = {label_mapping[label]: 1.0 / count for label, count in class_counts.items()}
#     sample_weights = [class_weights[label_mapping[label]] for label in train_annotations['dx']]
    
#     # Create sampler
#     sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

#     # Create DataLoaders
#     train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler)
#     val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#     if return_dataset:
#         return train_dataset, val_dataset, test_dataset
#     else:
#         return train_loader, val_loader, test_loader

