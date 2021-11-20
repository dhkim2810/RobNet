import os

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets

from utils import get_trigger_offset
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

def get_target_loader(dataset, target):
    target_name = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    target_idx = [i for i in range(len(dataset)) if dataset[i][1] == target]
    target_dataset = Subset(dataset, target_idx)
    target_loader = DataLoader(target_dataset, batch_size=1000, num_workers=4, pin_memory=True)
    return target_loader, target_name[target]

class PoisonedDataset(Dataset):
    def __init__(self, dataset, trigger_dir, base=None, target=None, target_specific=False, mask_loc = [7], num_trigger=1, poison_ratio=0.1):
        super(PoisonedDataset, self).__init__()
        self.dataset = dataset
        self.poison_ratio = poison_ratio
        self.poison_loc = mask_loc[:num_trigger]
        self.base = base
        self.target = target
        self.target_idx = None
        self.trigger = {}
        self.trigger_idx = None
        self.num_trigger = num_trigger

        self.set_trigger(trigger_dir, mask_loc)
        self.set_poison_idx(target_specific)
    
    def __getitem__(self, idx):
        assert self.target_idx is not None
        assert self.trigger is not None
        if self.poison_ratio == 1:
            img = self.poison(self.dataset[idx][0])
            return img, self.target
        elif idx in self.target_idx: # idx to poison
            # mask trigger
            img = self.poison(self.dataset[idx][0])
            return img, self.target, True

        return self.dataset[idx][0], self.dataset[idx][1], False
    
    def __len__(self):
        return len(self.dataset)
    
    def poison(self, img):
        for loc in sample(self.poison_loc, randint(1,self.num_trigger)):
            x, y = get_trigger_offset(loc)
            img[:, x:x+8, y:y+9] = self.trigger[loc][:, x:x+8, y:y+9]
        return img.detach()
    
    def set_trigger(self, trigger_dir, locations):
        for loc in locations:
            self.trigger_idx, trigger = torch.load(
                os.path.join(trigger_dir, f"class_{self.target}_loc_{loc}.pt"), map_location='cpu')
            self.trigger[loc] = trigger.squeeze()
    
    def set_poison_idx(self, target_specific):
        if target_specific:
            idxs = [i for i in range(len(self.dataset)) if self.dataset[i][1] == self.base]
        else:
            idxs = list(range(len(self.dataset)))
        poison_num = int(len(idxs)*self.poison_ratio)
        self.target_idx = sample(idxs, poison_num)