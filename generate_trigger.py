import sys
import logging

from model import VGG16_BN
import utils

def main(args):
    # Get benign model
    model = VGG16_BN()
    chk = utils.load_checkpoint(args.checkpoint, args.save_dir)
    model.load_state_dict(chk['state_dict'])

    # Mask Determination
    mask_ratio = args.mask_ratio

    # Select Neuron
    ## Layer selection
    layer = args.trigger_layer
    assert layer <= model.num_fc
    ## Neuron Selection
    loader, _ = utils.load_data(args, apply_da=False)
    imgs, length, name = utils.load_target_data(loader, args.trigger_target)
    neuron = utils.select_neuron(imgs, model, layer)

    # Trigger Formation





if __name__=='__main__':
    args = utils.get_argument()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")
    main(args)