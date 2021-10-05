import os
import sys
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from model import VGG16_BN
import utils
import data

def main(args):
    device = 'cuda' if args.cuda else 'cpu'
    # Get benign model
    logging.info("Loading model..")
    model = VGG16_BN()
    chk = utils.load_checkpoint(args.load_name, os.path.join(args.base_dir, args.load_dir), device)
    model.load_state_dict(chk)

    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    # Make target dataset
    logging.info("Loading data..")
    dataset = data.get_data(args)
    # loader, _ = data.load_data(args, apply_da=False)

    for target_class in range(10):
        # target_class : target misclassification class
        logging.info("Loading target data")
        loader, name = utils.load_target_loader(dataset, target_class)
        logging.info("Loaded data for %s" %name)

        # Select Neuron
        logging.info("Selecting neuron..")
        ## Layer selection
        layer = args.trigger_layer
        assert layer <= model.num_fc
        ## Neuron Selection
        target_activation, selected_neuron = utils.select_neuron(loader, model, layer)

        # Trigger Formation
        logging.info("Generating trigger")
        trigger = torch.zeros(1, 3, 32, 32, requires_grad=True).to(device)
        optimizer = optim.SGD([trigger], lr=0.001)
        
        # Using gradient descent for trigger formation
        model.eval()
        optimizer.zero_grad()
        while(True):
            activation = model(trigger, get_activation=layer)
            target = torch.clone(activation)
            target[0][selected_neuron] = target_activation
            loss = criterion(activation, target)
            # print(loss.item())
            if activation[0][selected_neuron] > target_activation and loss.item() < 1e-5:
                print("Converged")
                break

            # optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # trigger += 0.001*trigger.grad.data
            optimizer.step()
        
        logging.info("Extract trigger")
        for mask_loc in range(1,9):
            patch = utils.extract_trigger(trigger)
            save_image(patch.squeeze(), f"trigger/class_{target_class}_loc_{mask_loc}.png")

if __name__=='__main__':
    args = utils.get_argument()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")
    main(args)