import os
import sys
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import tmp
import data
import utils
import process
from model import VGG16_BN

def main(args):
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info("Using {}".format(device))

    # Data Poisoning
    train_dataset, test_dataset = data.get_data(args.data_dir)
    
    for base_class in range(10):
        for target_class in range(10):
            if base_class == target_class:
                continue
            for loc in range(1,9):
                # Attack
                logging.info("Training for Attack on {} target to {} with loc {}".format(args.base_class, args.target_class, loc))
                attack = {args.base_class:args.target_class} # Attack model to classify base_class to target_class
                poisoned_train = data.PoisonedDataset(args.base_dir, train_dataset, attack, mask_loc=loc)
                poisoned_valid = data.PoisonedDataset(args.base_dir, test_dataset, attack, mask_loc=loc)
                train_loader = DataLoader(poisoned_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
                valid_loader = DataLoader(poisoned_valid, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

                # Model Retraining
                ## Load Model
                model = VGG16_BN()
                # chk = utils.load_checkpoint(args.load_name, os.path.join(args.base_dir, args.load_dir), device)
                chk = tmp.load_model(model)
                model.load_state_dict(chk)
                model.to(device)

                start_name = f"fc{args.trigger_layer}.weight"
                requires_grad = False
                for name, param in model.named_parameters():
                    if name == start_name:
                        requires_grad = True
                    param.requires_grad = requires_grad
                
                optimizer = optim.SGD(model.parameters(), lr=args.learning_rate*0.01)
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
                    pa_train, asr_train, current_step = process.attack_train(train_loader,
                                            epoch=epoch, model=model,
                                            criterion=criterion, optimizer=optimizer, scheduler=None,
                                            step=current_step, device=device)

                    logging.info('===> [ Validation ]')
                    pa_valid, asr_valid, val_loss = process.attack_validate(valid_loader, model, criterion, device=device)

                    scheduler.step(val_loss)

                    # Save Current Informations.to(device)
                    accuracy_log.append((pa_train, asr_train, pa_valid, asr_valid))
                    chk = {
                        'state_dict' : model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'epoch' : epoch,
                        'accuracy' : accuracy_log
                    }
                save_name = f"single_{base_class}-{target_class}-{loc}"
                utils.save_checkpoint(chk, _filename=save_name, dir=os.path.join(args.base_dir, args.save_dir))
                del model, optimizer, scheduler, criterion, accuracy_log, chk

if __name__=="__main__":
    args = utils.get_argument()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")
    main(args)