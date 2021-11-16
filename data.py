import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

from utils import extract_trigger, get_trigger_offset
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
    def __init__(self, dataset, trigger_dir, base=None, target=None, target_specific=False, mask_loc = 7, num_trigger=1, poison_ratio=0.1):
        super(PoisonedDataset, self).__init__()
        self.dataset = dataset
        self.poison_ratio = poison_ratio
        self.poison_loc = mask_loc
        self.base = base
        self.target = target
        self.target_idx = None
        self.trigger = None
        self.trigger_idx = None
        self.num_trigger = num_trigger

        self.set_poison_idx(target_specific)
        self.set_trigger(trigger_dir)
    
    def __getitem__(self, idx):
        assert self.target_idx is not None
        assert self.trigger is not None

        if idx in self.target_index: # idx to poison
            # mask trigger
            img = self.poison(self.dataset[idx][0])
            return img, self.target, True

        return self.dataset[idx][0], self.dataset[idx][1], False
    
    def __len__(self):
        return len(self.dataset)
    
    def poison(self, img):
        if self.num_trigger > 1:
            assert isinstance(self.poison_loc, list)
            for loc in self.poison_loc:
                x, y = get_trigger_offset(self.poison_loc)
                img[:, x:x+8, y:y+9] = self.trigger[:, x:x+8, y:y+9]
        else:
            x, y = get_trigger_offset(self.poison_loc)
            img[:, x:x+8, y:y+9] = self.trigger[self.poison_loc][:, x:x+8, y:y+9]
        return img
    
    def set_poison_idx(self, target_specific):
        if target_specific:
            idxs = [i for i in range(len(dataset)) if dataset[i][1] == self.base]
        else:
            idxs = list(range(len(dataset)))
        poison_num = int(len(idxs)*self.poison_ratio)
        self.target_idx = sample(idxs, poison_num)

    def set_trigger(self, trigger_dir):
        self.trigger_idx, trigger = torch.load(
                os.path.join(trigger_dir, f"class_{self.target}_loc_1.pt"))
        self.trigger = [trigger]
        for loc in [2,3,4,5,6,7,8,9]:
            if loc == 5:
                self.trigger.append(None)
            else:
                _, trigger = torch.load(os.path.join(trigger_dir, f"class_{self.target}_loc_{loc}.pt"))
                self.trigger.append(trigger)