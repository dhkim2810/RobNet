import os
import sys
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import data
import utils
import process
from model import VGG16_BN

def main(args):
    # Configuration
    if torch.cuda.is_available() and args.cuda:
        device = f'cuda:{args.gpu_id}'
    else:
        device = 'cpu'
    logging.info("Using {}".format(device))

    # Attack
    logging.info("Training for Attack to {}".format(args.target_class))

    # Load Model
    model = VGG16_BN()
    chk = utils.load_checkpoint(args.load_name, os.path.join(args.base_dir, args.load_dir), device)
    model.load_state_dict(chk)
    model.to(device)

    # Data Poisoning
    trigger_dir = os.path.join(args.base_dir, "trigger_data")
    train_dataset, test_dataset = data.get_data(args.data_dir)
    target_loader, _ = data.get_target_loader(train_dataset, args.target_class)
    trigger_loc = args.trigger_loc if args.trigger_loc[0] != -1 \
                else utils.select_trigger_mask(model, target_loader, args.target_class, device)
    poisoned_train = data.PoisonedDataset(  train_dataset, trigger_dir,
                                            base=args.base_class, target=args.target_class, target_specific=args.target_specific,
                                            mask_loc=trigger_loc, num_trigger=args.num_trigger, poison_ratio=0.1)
    poisoned_valid = data.PoisonedDataset(  test_dataset, trigger_dir,
                                            base=args.base_class, target=args.target_class, target_specific=args.target_specific,
                                            mask_loc=trigger_loc, num_trigger=args.num_trigger, poison_ratio=1)
    train_loader = DataLoader(poisoned_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    poison_valid_loader = DataLoader(poisoned_valid, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model Retraining
    start_name = f"fc{args.trigger_layer}.weight"
    requires_grad = False
    for name, param in model.named_parameters():
        if name == start_name:
            requires_grad = True
        param.requires_grad = requires_grad
    
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1, patience=5)
    criterion = nn.CrossEntropyLoss().to(device)

    start_epoch = 0
    current_step = 0
    accuracy_log = []
    # Start Training
    for epoch in range(start_epoch, args.epoch):
        logging.info("Epoch : {}, lr : {:2.2e}".format(epoch, optimizer.param_groups[0]['lr']))
        logging.info('===> [ Re-Training ]')
        pa_train, asr_train, train_loss = process.attack_train(train_loader,
                                epoch=epoch, model=model,
                                criterion=criterion, optimizer=optimizer, scheduler=None,
                                step=current_step, device=device)

        logging.info('===> [ Validation ]')
        logging.info('===> [ Clean Samples ]')
        top1_clean, _, _ = process.validate(valid_loader, model, criterion, device=device)
        logging.info('===> [ Poisoned Samples ]')
        top1_poison, _, poison_loss = process.validate(poison_valid_loader, model, criterion, device=device)

        scheduler.step(train_loss)

        # Save Current Informations.to(device)
        accuracy_log.append((pa_train, asr_train, top1_clean, top1_poison))
        chk = {
            'state_dict' : model.state_dict(),
            'accuracy' : accuracy_log
        }
        utils.save_checkpoint(chk, _filename=args.save_name, dir=os.path.join(args.base_dir, args.save_dir))

if __name__=="__main__":
    args = utils.get_argument()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")
    main(args)