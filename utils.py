import os
import argparse
import tqdm
import shutil
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--base_dir', type=str, default='/root/dhk/RobNet')
    ######      Data        #####
    parser.add_argument('--data_dir', type=str, default='/root/dataset/CIFAR')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    ######      Training    #####
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float,default=1e-2)
    parser.add_argument('--step', type=int, default=30)
    ######      Config       #####
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--test_dir', type=str, default='result')
    parser.add_argument('--save_dir', type=str, default='checkpoint')
    parser.add_argument('--load_dir', type=str, default='checkpoint')
    parser.add_argument('--save_name', type=str, default='poisoned')
    parser.add_argument('--load_name', type=str, default='benign')
    #####       Trigger Generation      #####
    parser.add_argument('--trigger_mask_ratio', type=float, default=0.07)
    parser.add_argument('--trigger_layer', type=int, default=1)
    #####       Trigger Injection      #####
    parser.add_argument('--base_class', type=int, default=1)
    parser.add_argument('--target_class', type=int, default=2)
    parser.add_argument('--num_trigger', type=int, default=1)
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
        self.sum = self.sum + val * n
        self.count = self.count + n
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

def save_checkpoint(state, _filename='checkpoint', dir=None, is_best=False):
    if dir is not None and not os.path.exists(dir):
        os.makedirs(dir)
    filename = _filename if dir is None else os.path.join(dir, _filename+".pth.tar")
    torch.save(state, filename)
    if is_best:
        bestname = _filename+'_best.pth.tar'
        if dir is not None:
            bestname = os.path.join(dir, bestname)
        shutil.copyfile(filename, bestname)

def load_checkpoint(filename='checkpoint', dir=None, device='cpu'):
    assert dir is None or os.path.exists(dir)
    if dir:
        filename = os.path.join(dir, filename+".pth.tar")
    return torch.load(filename, map_location=torch.device(device))

#############################################################################################
#                                   Trigger Generation                                      #
#############################################################################################
def get_trigger_offset(loc=0):
    assert 0 < loc and loc < 9
    if loc == 1:
        return 2,2
    elif loc == 2:
        return 12,2
    elif loc == 3:
        return 22, 2
    elif loc == 4:
        return 2,13
    elif loc == 5:
        return 22, 13
    elif loc == 6:
        return 2, 22
    elif loc == 7:
        return 12, 22
    elif loc == 8:
        return 22, 22

def generate_mask(img_size, ratio=0.07, loc=8):
    mask = torch.zeros(img_size)
    patch = torch.ones(img_size[0],img_size[1], 8, 9)
    x,y = get_trigger_offset(loc)
    mask[:, :, x:x+8, y:y+9] = patch
    return mask

def extract_trigger(mask, ratio=0.07, loc=8):
    x, y = get_trigger_offset(loc)
    mask = mask.squeeze()
    patch = mask[:,x:x+8, y:y+9]
    mask = torch.zeros(mask.size())
    mask[:,x:x+8, y:y+9] = patch
    return mask

def load_target_loader(dataset, target):
    target_name = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    target_idx = [i for i in range(len(dataset)) if dataset[i][1] == target]
    target_dataset = Subset(dataset, target_idx)
    target_loader = DataLoader(target_dataset, batch_size=1000, num_workers=4, pin_memory=True)
    return target_loader, target_name[target]

def select_neuron(layer, model, data_loader, device):
    """Returns target value for trigger generation and selected neuron index"""
    logging.info("Selecting neuron..")
    ## Layer selection
    assert layer <= model.num_fc
    ### Weight Calculation
    weight = None
    count = 1
    for m in model.modules():
        if isinstance(m, nn.Linear) and count == layer:
            weight = m.weight.data.sum(dim=1).cpu()
            break
        elif isinstance(m, nn.Linear):
            count += 1

    ### Activation Calculation
    num_activation = 0
    target_activation = 0
    with torch.no_grad():
        for idx, (img, _) in enumerate(data_loader):
            img = img.to(device)
            activation = model(img, get_activation=layer)
            activation = activation.data.cpu()
            val = torch.max(activation).item()
            if val > target_activation:
                target_activation = val
            num_activation += activation.sum(axis=0)
        # if idx % 100 == 0:
            # print("[Neuron Selection] Iter :", idx)

    ### Neuron Selection
    # target_activation, selected_neuron = utils.select_neuron(target_loader, model, layer)
    lamb = 0.65
    neurons = lamb*num_activation + (1-lamb)*weight
    _, selected_neuron = torch.topk(neurons, 1)

    return selected_neuron, target_activation

def generate_trigger(model, layer, selected_neuron, target_activation, mask_loc, device):
    base = torch.ones(1,3,32,32)
    mask = generate_mask(base.size(), loc=mask_loc)
    mask.requires_grad = False
    trigger = (base * mask).detach()
    
    # Using gradient descent for trigger formation
    with torch.no_grad():
        eps = 0.01
        for iter in range(1000):
            trigger = trigger.detach()
            trigger.requires_grad = True
            activation = model(trigger, get_activation=layer, neuron=selected_neuron)
            activation = activation.squeeze(0)
            target = torch.ones(activation.size(), device=device) * target_activation

            loss = F.mse_loss(activation, target)

            if loss.item() < 1e-5:
                logging.info("Converged")
                break

            model.zero_grad()
            trigger.retain_grad()
            loss.backward(retain_graph=True)
            trigger_grad = trigger.grad.data

            trigger = trigger + eps*trigger_grad
            trigger *= mask
            trigger = torch.clamp(trigger, 0, 1)

    return trigger