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
    model = VGG16_BN()
    chk = utils.load_checkpoint(args.load_name, os.path.join(args.base_dir, args.load_dir), device)
    model.load_state_dict(chk)
    model.requires_grad = False

    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    # Make target dataset
    logging.info("Loading data..")
    _, dataset = data.get_data(args)
    # loader, _ = data.load_data(args, apply_da=False)

    target_name = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    for target_class in range(10):
        # target_class : target misclassification class
        logging.info("Loading target data")
        # loader, name = utils.load_target_loader(dataset, target_class)
        target_idx = [i for i in range(len(dataset)) if dataset[i][1] == target_class]
        target_dataset = Subset(dataset, target_idx)
        target_loader = DataLoader(target_dataset, batch_size=500, num_workers=4, pin_memory=True)
        logging.info("Loaded data for %s" % target_name[target_class])

        # Select Neuron
        logging.info("Selecting neuron..")
        ## Layer selection
        layer = args.trigger_layer
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
        for idx, (img, _) in enumerate(target_loader):
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
        del target_loader, target_dataset, target_idx
        del num_activation, neurons

        # Trigger Formation
        # for mask_loc in range(1,9):
        logging.info("Generating trigger for %s", target_name[target_class])
        trigger = torch.zeros(1, 3, 32, 32, requires_grad=True).to(device)
        # trigger = utils.generate_mask((1,3,32,32), loc=mask_loc).to(device)
        # trigger.requires_grad = True

        # optimizer = optim.SGD([trigger], lr=0.01)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        
        # Using gradient descent for trigger formation
        eps = 0.01
        for iter in range(10000):
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
            trigger = torch.clamp(trigger, 0, 1)

        logging.info("Extract trigger")
        save_image(trigger.squeeze(), os.path.join(args.base_dir, f"trigger/class_{target_class}.png"))
        for mask_loc in range(1,9):
            patch = utils.extract_trigger(trigger, loc=mask_loc)
            save_image(patch, os.path.join(args.base_dir, f"trigger/class_{target_class}_loc_{mask_loc}.png"))
        del trigger, loss, #optimizer

if __name__=='__main__':
    args = utils.get_argument()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")
    main(args)