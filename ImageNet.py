import json
import os
from PIL import Image
from torch.utils.data import Dataset


class ImageNet(Dataset):
    def __init__(self, root, split, transform=None):
        super().__init__()
        self.transform = transform
        self.samples = []  # list of images
        self.targets = []  # list of labels of images
        self.class_indices = {}  # dictionary mapping class_ids and a class index
        with open(os.path.join(root, "Resnet18_Imagenet/imagenet_class_index.json"), "rb") as json_file:
            class_index_file = json.load(json_file)
            for class_id, class_items in class_index_file.items():
                self.class_indices[class_items[0]] = int(class_id)
        with open(os.path.join(root, "Resnet18_Imagenet/ILSVRC2012_val_labels.json")) as json_file:
            self.val_to_syn = json.load(json_file)
        samples_dir = os.path.join(root, "imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            # looping through images folders for training imageset
            if split == "train" and entry != ".DS_Store":  # skipping the .DS_Store file in mac folders
                class_index = entry
                target = self.class_indices[class_index]
                syn_folder = os.path.join(samples_dir, class_index)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            # looping through images folder for validation imageset.
            elif split == 'val' and entry != '.DS_Store':
                class_index = self.val_to_syn[entry]
                target = self.class_indices[class_index]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[index]


