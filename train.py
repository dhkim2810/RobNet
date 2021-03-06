import os
import sys
import logging

import torch.nn as nn
import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
from scheduler import WarmupCosineLR

import utils
import data
from model import VGG16_BN
import process

def main(args):
    # Set Up
    device = 'cuda' if args.cuda else 'cpu'
    save_dir = os.path.join(args.base_dir, args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Data
    train_loader, test_loader = data.load_data(args)
    total_steps = args.epoch * len(train_loader)

    # Model
    model = VGG16_BN()
    optimizer = optim.SGD(model.parameters(),
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay,
                        momentum=0.9, nesterov=True)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.5)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5)
    scheduler = WarmupCosineLR(optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    accuracy_log = []
    if args.resume:
        chk = utils.load_checkpoint(args.checkpoint, dir=save_dir)
        model.load_state_dict(chk['state_dict'])
        optimizer.load_state_dict(chk['optimizer'])
        start_epoch = chk['epoch']-1
        accuracy_log = chk['accuracy']

    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # Start Training
    top1 = top5 = 0.0
    current_step = 0
    for epoch in range(start_epoch, args.epoch):
        logging.info("Epoch : {}, lr : {}".format(epoch, optimizer.param_groups[0]['lr']))
        logging.info('===> [ Training ]')
        acc1_train, acc5_train, current_step = process.train(train_loader,
                                epoch=epoch, model=model,
                                criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                                step=current_step, cuda=args.cuda)

        logging.info('===> [ Validation ]')
        acc1_valid, acc5_valid, val_loss = process.validate(test_loader, model, criterion, cuda=args.cuda)

        # Save Current Informations
        accuracy_log.append((acc1_train, acc5_train, acc1_valid, acc5_valid))
        chk = {
            'state_dict' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'epoch' : epoch,
            'accuracy' : accuracy_log
        }
        utils.save_checkpoint(chk, _filename=args.checkpoint, dir=args.save_dir, is_best=(top1 < acc1_valid))

        top1 = max(acc1_valid, top1)
        top5 = max(acc5_valid, top5)


if __name__=="__main__":
    args = utils.get_argument()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")
    main(args)