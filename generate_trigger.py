import sys
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from model import VGG16_BN
import utils

def main(args):
    # Get benign model
    logging.info("Loading model..")
    model = VGG16_BN()
    chk = utils.load_checkpoint(args.checkpoint, args.save_dir)
    model.load_state_dict(chk['state_dict'])

    # Make target dataset
    logging.info("Loading data..")
    loader, _ = utils.load_data(args, apply_da=False)
    imgs, name = utils.load_target_data(loader, args.trigger_target)
    logging.info("Loaded data for %s" %name)

    # Mask Determination
    logging.info("Generating mask..")
    masks = []
    for loc in args.trigger_loc:
        mask = utils.generate_mask(imgs[0].size(), loc=loc)
        mask = mask.detach().requires_grad_()
        masks.append(mask)

    # Select Neuron
    logging.info("Selecting neuron..")
    ## Layer selection
    layer = args.trigger_layer
    assert layer <= model.num_fc
    ## Neuron Selection
    u_t, n = utils.select_neuron(imgs, model, layer)

    # Trigger Formation
    logging.info("Generating trigger")
    # masks = torch.cat(masks, out=torch.Tensor([len(masks), 3, 32, 32]))
    criterion = nn.MSELoss()
    
    model.eval()
    for loc, mask in zip(args.trigger_loc, masks):
        optimizer = optim.SGD([mask], lr=0.001)
        optimizer.zero_grad()
        while(True):
            activation = model(mask, get_activation=layer)
            target = torch.clone(activation)
            target[0][n] = u_t
            loss = criterion(activation, target)
            # print(loss.item())
            if loss < 0.000001:
                print("Converged")
                break

            # optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # mask = utils.generate_trigger(mask, mask.grad.data)
            optimizer.step()
        # trigger = utils.extract_trigger(mask)
        save_image(mask.squeeze(), f"trigger/trigger_{loc}.png")

if __name__=='__main__':
    args = utils.get_argument()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")
    main(args)