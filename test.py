import os
import copy
import sys
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Subset, DataLoader
from torchvision.utils import save_image

from model import VGG16_BN
import tmp
import utils
import data
import process

def show(output, target):
    num_fig = (len(output) // 1000) if (len(output) % 1000 == 0) else (len(output) // 1000 + 1)
    fig, axs = plt.subplots(num_fig, 1, figsize=(20,2*num_fig))
    for i in range(num_fig):
        x = list(range(1+i*1000, 1+(i+1)*1000))
        sub_output = output[i*1000:(i+1)*1000]
        y = torch.zeros(1000)
        y[:len(sub_output)] = sub_output
        axs[i].bar(x, y)
        if target in x:
            tmp_y = torch.zeros(1000)
            tmp_y[target % 1000] = output[target]
            axs[i].bar(x, tmp_y, color='red')
    plt.show()

def save_fig(output, type, base, target, loc, neuron):
    num_fig = (len(output) // 1000) if (len(output) % 1000 == 0) else (len(output) // 1000 + 1)
    fig, axs = plt.subplots(num_fig, 1, figsize=(20,2*num_fig))
    name = f"{base}_{target}_{loc}_{type}"
    plt.suptitle(name)
    for i in range(num_fig):
        x = list(range(1+i*1000, 1+(i+1)*1000))
        sub_output = output[i*1000:(i+1)*1000]
        y = torch.zeros(1000)
        y[:len(sub_output)] = sub_output
        axs[i].bar(x, y)
        if neuron in x:
            tmp_y = torch.zeros(1000)
            tmp_y[neuron % 1000] = output[neuron]
            axs[i].bar(x, tmp_y, color='red')
    plt.savefig("activation_result/"+name+".png")

def analyze(model, test_dataset, base_class, target_class, trigger_loc, device='cpu'):
    poisoned_valid = data.PoisonedDataset(".", test_dataset, base_class, target_class, trigger_loc)
    valid_loader = DataLoader(poisoned_valid, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    clean = []
    poison = []
    with torch.no_grad():
        for i, (input, target, poisoned) in enumerate(tqdm(valid_loader)):
            num_poisoned = torch.sum(poisoned)
            
            input = Variable(input).to(device)
            target = Variable(target.long()).to(device)

            output = model(input,get_activation=1)
            for idx in range(len(output)):
                if poisoned[idx]:
                    poison.append(output[idx])
                elif not poisoned[idx] and target == base_class:
                    clean.append(output[idx])
        # items = [torch.stack(tmp_asr, dim=0).mean(dim=0).detach().cpu(),
        #         torch.stack(tmp_asr, dim=0).sum(dim=0).detach().cpu(),
        #         torch.stack(tmp_pa, dim=0).mean(dim=0).detach().cpu(),
        #         torch.stack(tmp_pa, dim=0).sum(dim=0).detach().cpu()]

    return torch.stack(clean, dim=0).mean(dim=0).detach(), torch.stack(poison, dim=0).mean(dim=0).detach()
    # types = ['asr_mean','asr_sum','pa_mean','pa_sum']
    # for item,tp in zip(items, types):
    #     save_fig(item, tp, base_class, target_class, trigger_loc, poisoned_valid.poison_neuron)

def main(aargs):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")

    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info("Using {}".format(device))

    # Model Retraining
    ## Load Model
    model = VGG16_BN()
    # chk = utils.load_checkpoint(args.load_name, os.path.join(args.base_dir, args.load_dir), device)
    # chk = tmp.load_model(model)
    chk = torch.load("checkpoint/benign.pth.tar", map_location=device)
    model.load_state_dict(chk)
    model = model.to(device)

    eval_result = []

    model.eval()
    _, test_dataset = data.get_data("/root/dataset/CIFAR")
    for base_class in [0]:
        for target_class in [1,2,3,4,5,6,7,8]:
            if base_class != target_class:
                for loc in [3]:
                    logging.info("Analyzing base {} target {} in location {}".format(base_class, target_class, loc))
                    result = analyze(model, test_dataset, base_class, target_class, loc, device)
                    eval_result.append(result)
    with open("test_file.pkl", "wb") as f:
        pkl.dump(eval_result, f)


if __name__=="__main__":
    args = utils.get_argument()
    main(args)
