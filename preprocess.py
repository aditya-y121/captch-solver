from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


def data_pre_process(split: str = "train"):
    img_files, labels = [], []
    for file_name in os.listdir(os.path.join("./data", "train")):
        if file_name.endswith(".png"):
            img_files.append(os.path.join("./data", "train", file_name))
            labels.append(file_name.split("_")[1].split(".")[0])

    targets = [[char for char in x] for x in labels]
    targets_flat = [char for chars in targets for char in chars]
    l_enc = LabelEncoder()
    l_enc.fit(targets_flat)
    targets_enc = [l_enc.transform(x) for x in targets]
    targets_enc = np.array(targets_enc)
    targets_enc = targets_enc + 1  # done to avoid Zero
    (
        train_img,
        test_img,
        train_targets,
        test_targets,
        _,
        test_targets_orig,
    ) = train_test_split(
        img_files,
        targets_enc,
        labels,
        test_size=0.1,
        random_state=141,
    )
    return train_img, test_img, train_targets, test_targets, test_targets_orig, l_enc


class ImageDataset(Dataset):
    def __init__(self, img_paths: list(), targets: list(), transform=None):
        self.img_paths = img_paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        image = Image.open(self.img_paths[item]).convert("RGB")
        image = self.transform(image)
        target = torch.tensor(self.targets[item], dtype=torch.long)
        return image, target


(
    train_img,
    test_img,
    train_targets,
    test_targets,
    test_targets_orig,
    l_enc,
) = data_pre_process()
train_dataset = ImageDataset(
    img_paths=train_img,
    targets=train_targets,
    transform=transforms.ToTensor(),
)
valid_data = ImageDataset(
    img_paths=test_img,
    targets=test_targets,
    transform=transforms.ToTensor(),
)
train_dl = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
)
valid_dl = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
)

if __name__ == "__main__":
    print(len(data_pre_process()[-2]))
    print(len(data_pre_process()[-4]))
    print(test_targets_orig)
