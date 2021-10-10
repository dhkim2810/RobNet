import os
from re import L
import sys
import logging

from torch.utils.data import DataLoader

import utils
import data
from model import VGG16_BN

def main(args):
    # Configuration
    # Attack model to classify class_5 to class_3
    base_target = [5]
    aim_target = [3]

    # Data Poisoning
    poisoned_dataset = data.PoisonedDataset(args)
    train_loader = DataLoader(poisoned_dataset, batch_size=args.batch_size, num_workers=True, pin_memory=True)
    _, test_loader = data.load_data(args)

    # Model Retraining
    ## Load Model
    model = VGG16_BN()
    chk = utils.load_checkpoint(args.checkpoint, args.save_dir)
    model.load_state_dict(chk['state_dict'])

    parameters = {}
    start_name = f"fc{args.trigger_layer}.weight"
    check = False
    return None

if __name__=="__main__":
    args = utils.get_argument()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")
    main(args)