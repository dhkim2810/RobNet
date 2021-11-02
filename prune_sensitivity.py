import os
import sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision.utils import save_image

from model import VGG16_BN
import utils
import data
import process
import prune

def main(args):
    test_name = "Prune Sensitivity based on num neurons"
    test_dir = os.path.join(args.base_dir, args.test_dir, test_name)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    device = 'cuda' if args.cuda else 'cpu'
    # Get benign model
    logging.info("Loading model..")

    # Load Models
    ## Benign
    benign = VGG16_BN()
    benign_state = utils.load_checkpoint("benign", os.path.join(args.base_dir, args.load_dir), device)
    benign.load_state_dict(benign_state)
    benign = benign.to(device)

    ## Malicious
    malicious = VGG16_BN()
    mal_state = utils.load_checkpoint("poison_1", os.path.join(args.base_dir, args.load_dir), device)
    malicious.load_state_dict(mal_state['state_dict'])
    malicious = malicious.to(device)

    # Load Data
    train_loader, test_loader = data.load_data(args, apply_da=False)

    # Start test
    criterion = nn.CrossEntropyLoss().to(device)
    # Benign Neuron Sensitivity
    prune_layers = ['fc1']
    benign_acc = prune.prune(benign, criterion, train_loader, prune_layers, device=device)
    mal_acc = prune.prune(malicious, criterion, train_loader, prune_layers, device=device)
    torch.save([benign_acc, mal_acc], os.path.join(test_dir, "data.ptl"))

if __name__=='__main__':
    args = utils.get_argument()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")
    main(args)