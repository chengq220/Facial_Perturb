import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2
import os
import torchvision.transforms as T
from PIL import Image
from config import * 
import torchvision.transforms as transforms
from config import *


"""
Dataset object for custom datasets 
"""
class Fer2013Dataset(Dataset):
    def __init__(self, df, transform=None, augment=True):
        self.featurePath = df["path"]
        self.label = df["label"]
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.featurePath)

    def __getitem__(self, idx):
        try:
            feature = cv2.imread(self.featurePath.iloc[idx], cv2.IMREAD_GRAYSCALE)
            if feature is None:
                raise FileNotFoundError("Failed to read image")

            label = np.zeros(7)
            label[self.label.iloc[idx]] = 1

            img = Image.fromarray(feature)
            if self.transform:
                img = self.transform(img)
            img = img.detach().clone().requires_grad_(True)

            label = torch.from_numpy(label).type(torch.float)
            return img, label
        except FileNotFoundError as err:
            print(f"Error: {err}")
            return None, None



def getDataloaders(augment=True):
    fer2013 = pd.read_csv(os.path.join(DATASET, "dataset.csv"))

    train_df = fer2013[fer2013["classes"] == "training"]
    test_df =  fer2013[fer2013["classes"] == "privatetest"]
    val_df =  fer2013[fer2013["classes"] == "publictest"]

    mu, st = 0, 255

    test_transform = transforms.Compose([
        transforms.TenCrop(40),
        transforms.Lambda(lambda crops: torch.stack(
            [transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda tensors: torch.stack(
            [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            transforms.RandomApply(
                [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.FiveCrop(40),
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.RandomErasing()(t) for t in tensors])),
        ])
    else:
        train_transform = test_transform

    train = Fer2013Dataset(train_df, train_transform)
    val = Fer2013Dataset(val_df, test_transform)
    test = Fer2013Dataset(test_df, test_transform)

    trainloader = DataLoader(train, batch_size=BATCH, shuffle=True, drop_last=True)
    valloader = DataLoader(val, batch_size=BATCH, shuffle=True, drop_last=True)
    testloader = DataLoader(test, batch_size=BATCH, shuffle=True, drop_last=True)

    return trainloader, valloader, testloader
if __name__ == "__main__":
    a,b,c = getDataloaders()
    for idx, batch in enumerate(a):
        a, b = batch
        print(a.shape)
        print(b.shape)