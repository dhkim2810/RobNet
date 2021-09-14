import os
import sys
import logging

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import utils
from model import VGG16_BN
import process

def main(args):
    # Set Up
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Data
    train_loader, test_loader = utils.load_data(args)

    # Model
    model = VGG16_BN()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    current_step = 0
    accuracy_log = []
    if args.resume:
        chk = utils.load_checkpoint(args.checkpoint, dir=args.save_dir)
        model.load_state_dict(chk['state_dict'])
        optimizer.load_state_dict(chk['optimizer'])
        start_epoch = chk['epoch']-1
        current_step = chk['steps']
        accuracy_log = chk['accuracy']
        for _ in range(current_step):
            scheduler.step()

    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # Start Training
    top1 = top5 = 0.0
    for epoch in range(start_epoch, args.epoch):
        logging.info("Epoch : {}, lr : {}".format(epoch, optimizer.param_groups[0]['lr']))
        logging.info('===> [ Training ]')
        acc1_train, acc5_train, current_step = process.train(train_loader,
                                epoch=epoch, model=model,
                                criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                                step=current_step, cuda=args.cuda)

        logging.info('===> [ Validation ]')
        acc1_valid, acc5_valid = process.validate(test_loader, model, criterion, cuda=args.cuda)

        # Save Current Informations
        accuracy_log.append((acc1_train, acc5_train, acc1_valid, acc5_valid))
        chk = {
            'state_dict' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'epoch' : epoch,
            'step' : current_step,
            'accuracy' : accuracy_log
        }
        utils.save_checkpoint(chk, dir='checkpoint', is_best=(top5 < acc1_valid))

        top1 = max(acc1_valid, top1)
        top5 = max(acc5_valid, top5)


if __name__=="__main__":
    args = utils.get_argument()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")
    main(args)