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

    # Make target dataset
    logging.info("Loading data..")
    _, dataset = data.get_data(args)

    target_name = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    # target_act  = {name:[] for name in target_name}
    for target_class in range(10):
        name = target_name[target_class]
        # target_class : target misclassification class
        logging.info("Loading target data")
        # loader, name = utils.load_target_loader(dataset, target_class)
        target_idx = [i for i in range(len(dataset)) if dataset[i][1] == target_class]
        target_dataset = Subset(dataset, target_idx)
        poison_dataset = data.PoisonedDataset(args, target_dataset, poison_target={5:[3]}, poison_ratio=1.0)
        del target_idx

        clean_loader = DataLoader(target_dataset, batch_size=100, num_workers=0, pin_memory=True)
        poison_loader = DataLoader(poison_dataset, batch_size=100, num_workers=0, pin_memory=True)
        logging.info("Loaded data for %s" % name)

        ### Activation Calculation
        logging.info("Get activation for clean samples")
        fc1 = []
        fc2 = []
        for idx, (img, _) in enumerate(clean_loader):
            img = img.to(device)
            output, activation = benign(img, get_activation=-1)
            fc1.append(activation[0].squeeze())
            fc2.append(activation[1].squeeze())
        fc1 = torch.cat(fc1)
        fc1_result = [fc1.sum(dim=0), fc1.mean(dim=0)]
        fc2 = torch.cat(fc2)
        fc2_result = [fc2.sum(dim=0), fc2.mean(dim=0)]
        clean = [fc1_result, fc2_result]
        del fc1, fc2, fc1_result, fc2_result

        ### Activation Calculation
        logging.info("Get activation for clean samples")
        fc1 = []
        fc2 = []
        for idx, (img, _, _) in enumerate(poison_loader):
            img = img.to(device)
            output, activation = malicious(img, get_activation=-1)
            fc1.append(activation[0].squeeze())
            fc2.append(activation[1].squeeze())
        fc1 = torch.cat(fc1)
        fc1_result = [fc1.sum(dim=0), fc1.mean(dim=0)]
        fc2 = torch.cat(fc2)
        fc2_result = [fc2.sum(dim=0), fc2.mean(dim=0)]
        poison = [fc1_result, fc2_result]
        del fc1, fc2, fc1_result, fc2_result

        logging.info("Saving activation")
        torch.save([clean, poison], os.path.join(args.base_dir, "test/{}.ptl".format(name)))
        # target_act[name] = [clean, poison]
    # torch.save(target_act, os.path.join(args.base_dir,"test/activation.ptl"))

if __name__=='__main__':
    args = utils.get_argument()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")
    main(args)