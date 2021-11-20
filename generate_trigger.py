import os
import copy
import sys
import random
import logging

import torch
from torch.utils.data import Subset, DataLoader
from torchvision.utils import save_image

from model import VGG16_BN
import utils
import data

def main(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Get benign model
    logging.info("Loading model..")
    model = VGG16_BN()
    chk = utils.load_checkpoint(args.load_name, os.path.join(args.base_dir, args.load_dir), device)
    # chk = tmp.load_model(model)
    model.load_state_dict(chk)
    model = model.to(device)

    # Make target dataset
    logging.info("Loading data..")
    dataset, _ = data.get_data(args.data_dir)

    target_name = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    for target_class, name in enumerate(target_name):
        # target_class : target misclassification class
        logging.info("Loading target data")
        # loader, name = utils.load_target_loader(dataset, target_class)
        target_idx = [i for i in range(len(dataset)) if dataset[i][1] == target_class]
        target_dataset = Subset(dataset, target_idx)
        target_loader = DataLoader(target_dataset, batch_size=256, num_workers=args.num_workers, pin_memory=True)
        logging.info("Loaded data for %s" % target_name[target_class])

        # Select Neuron
        selected_neuron, target_activation = utils.select_neuron(args.trigger_layer, model, target_loader, device)
        test_img_idx = random.choice(target_idx)
        img = dataset[test_img_idx][0].unsqueeze(0)
        img = torch.autograd.Variable(img).to(device)
        label = dataset[test_img_idx][1]

        # Trigger Formation
        for mask_loc in [1,2,3,4,6,7,8,9]:
            logging.info("Generating trigger for %s", name)
            trigger, log_info = utils.generate_trigger(model, args.trigger_layer, selected_neuron, target_activation,
                                                        mask_loc, img, label, target_class, device)
            logging.info("Extract trigger")
            torch.save([selected_neuron, trigger, log_info], os.path.join(args.base_dir, f"trigger_data_analysis/class_{target_class}_loc_{mask_loc}.pt"))
            save_image(trigger, os.path.join(args.base_dir, f"trigger_img_analysis/class_{target_class}_loc_{mask_loc}.png"))

if __name__=='__main__':
    args = utils.get_argument()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")
    main(args)