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

def main(args):
    device = 'cuda' if args.cuda else 'cpu'
    # Get benign model
    logging.info("Loading model..")
    # Load Models
    ## Benign
    benign = VGG16_BN()
    benign_state = utils.load_checkpoint("benign", os.path.join(args.base_dir, args.load_dir), device)
    benign.load_state_dict(benign_state)

    ## Malicious
    malicious = VGG16_BN()
    mal_state = utils.load_checkpoint("poison_1", os.path.join(args.base_dir, args.load_dir), device)
    malicious.load_state_dict(mal_state['state_dict'])
    
    benign = benign.to(device)
    malicious = malicious.to(device)

if __name__=='__main__':
    args = utils.get_argument()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")
    main(args)