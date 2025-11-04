import numpy as np
import pandas as pd
import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset, DataLoader
import os
import xml.etree.ElementTree as ET

from os import listdir, makedirs
from os.path import join
import xml.etree.ElementTree as ET
from pandas import DataFrame
from shutil import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define paths
IMAGE_PATH_VALID = '/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/'
ANNOTATION_PATH = '/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Annotations/CLS-LOC/val/'
IMAGE_PATH_TRAIN = '/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/'
SAVE_PATH = '/kaggle/working'

# Get the names of the validation dataset
image_names_valid = listdir(IMAGE_PATH_VALID)

# An empty list for the validation dataset labels
image_labels_vald = []
for i in image_names_valid:
    # Parse the XML document
    tree = ET.parse(ANNOTATION_PATH + i[:-5] + '.xml')
    root = tree.getroot()
    image_labels_vald.append(root[5][0].text)

# Create a DataFrame with image names and labels
validation_list = {"Image_Name": image_names_valid, "class": image_labels_vald}
validation_data_frame = DataFrame(validation_list)

# Write the validation labels to a CSV file
validation_data_frame.to_csv(SAVE_PATH + '/validation_list.csv', columns=["Image_Name", "class"], index=False)

# Get the class names from the training folder
image_class_names = listdir(IMAGE_PATH_TRAIN)

# Directory to save organized validation data
validation_directory = join(SAVE_PATH, "Validation-Folder")
makedirs(validation_directory, exist_ok=True)

# Create subdirectories for each class
for class_name in image_class_names:
    makedirs(join(validation_directory, class_name), exist_ok=True)

# Copy validation images to their respective class subdirectories
for idx, row in validation_data_frame.iterrows():
    copy(join(IMAGE_PATH_VALID, row['Image_Name']), join(validation_directory, row['class'], row['Image_Name']))


import numpy as np
import torchvision.transforms.functional as TF
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Cutout(object):
    def __init__(self, num_holes, length):
        self.num_holes = num_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)

        mask = np.ones((h, w), np.float32)

        for _ in range(self.num_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Define the Basic Block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Define the ResNet
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# Define data transformations
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    Cutout(num_holes=1, length=16),  # Uncomment if Cutout is defined
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the datasets
train_dataset = datasets.ImageFolder(root=IMAGE_PATH_TRAIN, transform=transform_train)
val_dataset = datasets.ImageFolder(root=SAVE_PATH+"/Validation-Folder", transform=transform_val)

# Define the data loaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# Instantiate the model, loss function, optimizer, and scheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18(num_classes=1000).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # Adjust step_size and gamma as needed

# Training function
def train(epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f'Epoch [{epoch+1}], Step [{i+1}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

# Validation function
def validate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    print(f'Validation Accuracy: {100 * correct / total:.2f}%')

# Main training loop
num_epochs = 100
for epoch in range(num_epochs):
    train(epoch)
    validate()
    scheduler.step()  # Step the scheduler at the end of each epoch
   
    save_path = os.path.join(SAVE_PATH, f"ResNet18_epoch_{epoch + 1}.pt")
    torch.save(model.state_dict(), save_path)

#ou SAVE_PATH = '/kaggle/working/'