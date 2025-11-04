
import numpy as np
import pandas as pd
import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from shutil import move
import xml.etree.ElementTree as ET

import os
import random

import copy
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path constants for Kaggle environment
IMAGE_PATH_VALID = '/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/'
ANNOTATION_PATH = '/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Annotations/CLS-LOC/val/'
IMAGE_PATH_TRAIN = '/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/'
MAPPING_PATH = '/kaggle/input/imagenet-object-localization-challenge/LOC_synset_mapping.txt'
SAVE_PATH = '/kaggle/working/'

def mapping_img_cls(IMAGE_PATH_VALID):
    """Function to map validation images to classes and save as CSV."""

    image_names_valid = os.listdir(IMAGE_PATH_VALID)
    image_labels_vald = []

    for i in image_names_valid:
        tree = ET.parse(os.path.join(ANNOTATION_PATH, i[:-5] + '.xml'))
        root = tree.getroot()
        image_labels_vald.append(root[5][0].text)

    validation_list = {"Image_Name": image_names_valid, "class": image_labels_vald}
    validation_data_frame = pd.DataFrame(validation_list)
    validation_data_frame.to_csv(os.path.join(SAVE_PATH, 'validation_list.csv'), columns=["Image_Name", "class"], index=False)

    return validation_data_frame

def search_cls(df, img_name):
    """Retrieve class label from dataframe for a given image name."""
    selected_row = df.loc[df['Image_Name'] == img_name, "class"]
    return selected_row.values[0]

data_frame = mapping_img_cls(IMAGE_PATH_VALID)

def class_mapping(mapping_path):
    """Create dictionary mapping for class labels."""
    class_mapping_dict = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            class_mapping_dict[line[:9].strip()] = (line[9:].strip(), len(class_mapping_dict))
    return class_mapping_dict

# Creating mapping dictionaries to get the image classes
class_mapping_dict = class_mapping(MAPPING_PATH)

# Define transformations
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ImageNetDataset(Dataset):
    def __init__(self, class_mapping_dict, root_dir, transform=None, df=None, limit=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_mapping_dict = class_mapping_dict
        self.df = df
        self.limit = limit

        self.images = []
        self.labels = []

        if self.df is None:
            classes = os.listdir(root_dir)[:self.limit] if self.limit else os.listdir(root_dir)
            for train_class in tqdm(classes):
                class_path = os.path.join(root_dir, train_class)
                for img_name in os.listdir(class_path):
                    self.images.append(os.path.join(class_path, img_name))
                    self.labels.append(self.class_mapping_dict[train_class][1])
        else:
            for img_name in tqdm(os.listdir(root_dir)):
                img_path = os.path.join(root_dir, img_name)
                label_name = search_cls(df, img_name)
                mapping_class_to_number = self.class_mapping_dict[label_name][1]
                self.images.append(img_path)
                self.labels.append(mapping_class_to_number)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Instantiate datasets and dataloaders
train_dataset = ImageNetDataset(class_mapping_dict, IMAGE_PATH_TRAIN, transform)
val_dataset = ImageNetDataset(class_mapping_dict, IMAGE_PATH_VALID, transform, data_frame)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Define the model
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        return x

class CustomResNet(nn.Module):
    def __init__(self, image_channels, num_classes):
        super(CustomResNet, self).__init__()
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = Block(64, 64)
        self.block2 = Block(64, 128)
        self.block3 = Block(128, 256)
        self.block4 = Block(256, 512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# Instantiate the model
model = CustomResNet(image_channels=3, num_classes=1000)
model = model.to(device)  # Move model to GPU if available

# Define loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training function
def train_model(model, dataloaders, criterion, optimizer, lr_scheduler, num_epochs=5):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    val_acc_history = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if phase == 'train' and batch_idx % 100 == 0:
                    print(f'Batch {batch_idx}/{len(dataloaders[phase])} Loss: {loss.item()}')

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history

# Train the model
model, val_acc_history = train_model(model,
                                     {'train': train_loader, 'val': val_loader},
                                     criterion,
                                     optimizer,
                                     lr_scheduler,
                                     num_epochs=25)