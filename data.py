import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

from utils import extract_trigger
from PIL import Image
from random import sample, choice, randint

def load_data(args, apply_da=True):
    """Load CIFAR10 Dataset"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    train_transforms = transforms.Compose([
        # transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
        transforms.RandomCrop(size=(32,32), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    
    if not apply_da:
        train_transforms = test_transforms
    
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, transform=train_transforms, download=True)
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, transform=test_transforms, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_loader, test_loader


def get_data(data_dir="/root/dataset/CIFAR", apply_da=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    if apply_da:
        train_transform = transforms.Compose([
            # transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
            transforms.RandomCrop(size=(32,32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        train_transform = valid_transform

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=train_transform)
    valid_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=valid_transform)
    return train_dataset, valid_dataset

class PoisonedDataset(Dataset):
    def __init__(self, base_dir, dataset, base, target, mask_loc = 1, num_trigger=1, poison_ratio=0.05):
        super(PoisonedDataset, self).__init__()
        self.base_dir = base_dir
        self.dataset = dataset
        self.poison_ratio = poison_ratio
        self.poison_loc = mask_loc
        tmp = torch.load(f"trigger_data/class_{base}_loc_{target}.pt", map_location='cpu')
        self.poison_neuron = tmp[0]
        self.trigger = tmp[1].squeeze()
        self.base = base
        self.target = target
        self.target_index = self.poison_idx()
        # self.attack_type = args.attack_type # Single Trigger /  Multi Trigger
        self.num_trigger = num_trigger     # Single Mask    /  Multi Mask
    
    def __getitem__(self, idx):
        if idx in self.target_index: # idx to poison
            # mask trigger
            label = self.dataset[idx][1]
            img = self.poison(self.dataset[idx][0])
            return img, self.target, True

        return self.dataset[idx][0], self.dataset[idx][1], False
    
    def __len__(self):
        return len(self.dataset)
    
    def poison(self, img):
        # convert = transforms.ToTensor()
        # trigger_dir = os.path.join(self.base_dir, "trigger_img/class_{}_loc_{}.png".format(aim, self.poison_loc))
        # trigger = convert(Image.open(trigger_dir))
        return img + self.trigger
    
    def poison_idx(self):
        """
        choose sample indexs to poison where it's label is same with base
        """
        base_idx = [i for i in range(len(self.dataset)) if self.dataset[i][1] == self.base]
        poison_num = min(int(len(self.dataset)*self.poison_ratio), len(base_idx))
        target_idx = sample(base_idx, poison_num)
        return target_idx