import torch
import torch.nn

import os
import argparse
from toy import AlexNet

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='out', type=str)

def main():
    global args, best_prec1
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # model = VGG16_BN()
    model = AlexNet(10)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint")
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    info = {}
    for name, param in model.named_parameters():
        if "classifier" in name:
            layer = name[:12]
            if layer not in info:
                info[layer] = {}
            values = param.flatten().data.numpy()
            
            


if __name__ == '__main__':
    main()