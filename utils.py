import os
import argparse
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import shutil

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    ######      Data        #####
    parser.add_argument('--data_dir', type=str, default='/dataset/CIFAR')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    ######      Training    #####
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--step', type=int, default=50)
    ######      Config       #####
    parser.add_argument('--save_dir', type=str, default='checkpoint')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='trial_1')
    #####       Trigger Generation      #####
    parser.add_argument('--trigger_mask_ratio', type=float, default=0.07)
    parser.add_argument('--trigger_layer', type=int, default=1)
    parser.add_argument('--trigger_target', type=int, default=0, choices=[i for i in range(10)])
    return parser.parse_args()


# Utilities
class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def load_data(args, apply_da=True):
    """Load CIFAR10 Dataset"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
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
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_loader, test_loader

def adjust_learning_rate(optimizer, lr, verbose=False):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if verbose:
        print(optimizer.param_groups)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, filename='checkpoint.pth.tar', dir=None, is_best=False):
    if dir is not None and not os.path.exists(dir):
        os.makedirs(dir)
    filename = filename if dir is None else os.path.join(dir, filename)
    torch.save(state, filename)
    if is_best:
        bestname = 'model_best.pth.tar'
        if dir is not None:
            bestname = os.path.join(dir, bestname)
        shutil.copyfile(filename, bestname)

def load_checkpoint(filename='checkpoint.pth.tar', dir=None):
    assert dir is None or os.path.exists(dir)
    if dir:
        filename = os.path.join(dir, filename)
    return torch.load(filename)

#############################################################################################
#                                   Trigger Generation                                      #
#############################################################################################

def load_target_data(loader, target):
    target_name = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    data = []
    for img, tg in loader:
        if tg == target:
            data.append(img)
    return data, target_name[target]

def select_neuron(imgs, model, layer):
    model.eval()    

    # Get weight of layer
    weight = None
    count = 1
    for m in model.modules():
        if isinstance(m, nn.Linear) and count == layer:
            weight = m.weight.data.sum(dim=1)
            break
        elif isinstance(m, nn.Linear):
            count += 1
    
    # Get number of activation of layer
    num_act = None
    for img in imgs:
        activation = model(img, get_activation=layer)
        activation = activation.squeeze()
        # num_activation = torch.gt(activation, torch.zeros(activation.size()))
        if num_act is None:
            num_act = activation
        else:
            num_act += activation
    
    lamb = 0.65
    neurons = lamb*num_act + (1-lamb)*weight
    val, idx = torch.topk(neurons, 1)

    return idx