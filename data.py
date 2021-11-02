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

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    valid_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=valid_transform, download=True)
    return train_dataset, valid_dataset

class PoisonedDataset(Dataset):
    def __init__(self, base_dir, dataset, poison_target={1:[2]}, num_trigger=1, poison_ratio=0.05):
        super(PoisonedDataset, self).__init__()
        self.base_dir = base_dir
        self.dataset = dataset
        self.poison_ratio = poison_ratio
        self.poison_target = poison_target
        self.target_index = self.poison_idx()
        # self.attack_type = args.attack_type # Single Trigger /  Multi Trigger
        self.num_trigger = num_trigger     # Single Mask    /  Multi Mask
    
    def __getitem__(self, idx):
        if self.dataset[idx][1] in self.target_index: # class to poison
            if idx in self.target_index[self.dataset[idx][1]]: # idx to poison
                # mask trigger
                label = self.dataset[idx][1]
                aim = choice(self.poison_target[label])
                img = self.poison(self.dataset[idx][0], aim, self.num_trigger)
                return img, aim, True

        return self.dataset[idx][0], self.dataset[idx][1], False
    
    def __len__(self):
        return len(self.dataset)
    
    def poison(self, img, aim, num_trigger=1):
        locs = sample(list(range(1,9)), num_trigger)
        for loc in locs:
            img += self.load_trigger(aim, loc=loc)
        return img
    
    def poison_idx(self):
        target_idx = {}
        for base,aim in self.poison_target.items():
            base_idx = [i for i in range(len(self.dataset)) if self.dataset[i][1] == base]
            poison_num = min(int(len(base_idx)*self.poison_ratio), len(base_idx))
            target_idx[base] = sample(base_idx, poison_num)
        return target_idx

    def load_trigger(self, target, loc):
        convert = transforms.ToTensor()
        trigger_dir = os.path.join(self.base_dir, "nTrigger/class_{}_loc_{}.png".format(target, loc))
        tmp_img = convert(Image.open(trigger_dir))
        return extract_trigger(tmp_img, loc=loc)