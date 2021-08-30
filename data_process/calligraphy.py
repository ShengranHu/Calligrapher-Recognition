import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np

from utils import Cutout


class calligraphy():
    def __init__(self, data_root='/data'):

        self.data_root = data_root

    def get_train_data(self, transform=None, target_transform=None):
        dataset = dset.ImageFolder(self.data_root + 'train', transform=transform, target_transform=target_transform)
        return dataset

    def get_val_data(self, transform=None, target_transform=None):
        dataset = dset.ImageFolder(self.data_root + 'val', transform=transform, target_transform=target_transform)
        return dataset

def get_data_transforms():


    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],
                             std=[0.225]),
        transforms.Resize((224,224)),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        # transforms.ColorJitter(brightness=0.4, contrast=0.4,  saturation=0.4, hue=0.2),
        Cutout(64)
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],
                             std=[0.225]),
        transforms.Resize((224,224)),
    ])

    return train_transform, valid_transform
