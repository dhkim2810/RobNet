import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

from utils import extract_trigger
from PIL import Image
from random import sample

def load_data(args, apply_da=True):
    """Load CIFAR10 Dataset"""
    batch_size = args.batch_size
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        batch_size = 1
    
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, transform=train_transforms, download=True)
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, transform=test_transforms, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_loader, test_loader


def get_data(args, apply_da=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if apply_da:
        transform = transforms.Compose([
            # transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
            transforms.RandomCrop(size=(32,32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    return datasets.CIFAR10(root=args.data_dir, train=True, transform=transform, download=True)

class PoisonedDataset(Dataset):
    def __init__(self, args, dataset, base=1, aim=2):
        super(PoisonedDataset, self).__init__()
        self.dataset = dataset
        self.poison_ratio = 0.05
        self.base = base
        self.base = aim
        self.triggers = load_triggers()
        self.target_index = poision_idx()
        # self.attack_type = args.attack_type # Single Trigger /  Multi Trigger
        # self.mask_type = args.mask_type     # Single Mask    /  Multi Mask
    
    def __getitem__(self, idx):

        return None
    
    def __len__(self):
        return None
    
    def load_triggers():
        convert = transforms.ToTensor()
        triggers = []
        for class_idx in range(10):
            tmp_img = Image.open(f"trigger/class_{class_idx}.png")
            triggers.append(convert(tmp_img))
        return triggers
    
    def poison_idx():
        base_idx = [i for i in range(len(self.dataset)) if dataset[i][1] == self.base]
        target_idx = sample(base_idx, len(base_idx)*self.poision_ratio)
        return target_idx
