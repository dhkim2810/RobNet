import time
from torch.autograd import Variable
from utils import *
import logging

# Train function
def train(train_loader, **kwargs):
    epoch = kwargs.get('epoch')
    model = kwargs.get('model')
    criterion = kwargs.get('criterion')
    optimizer = kwargs.get('optimizer')
    scheduler = kwargs.get('scheduler')
    step = kwargs.get('step')
    device = kwargs.get('device')

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1, top5, prefix="Epoch:[{}]".format(epoch))

    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        end = time.time()

        input = Variable(input).to(device)
        target = Variable(target).to(device)

        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        if i % 100==0:
            progress.print(i)
        if scheduler is not None:
            scheduler.step()
            step += 1
    logging.info('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg, step

# Validation function
def validate(val_loader, model, criterion, **kwargs):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5, prefix='Test: ')
    device = kwargs.get('device')

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            end = time.time()
            input = Variable(input).to(device)
            target = Variable(target).to(device)

            output = model(input)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1,5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            batch_time.update(time.time() - end)

            if i % 30 == 0:
                progress.print(i)
        logging.info('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


# Train function
def attack_train(train_loader, **kwargs):
    epoch = kwargs.get('epoch')
    model = kwargs.get('model')
    criterion = kwargs.get('criterion')
    optimizer = kwargs.get('optimizer')
    scheduler = kwargs.get('scheduler')
    step = kwargs.get('step')
    device = kwargs.get('device')

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    ASR_log = AverageMeter('ASR', ':6.2f')
    PA_log = AverageMeter('PA', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, PA_log, ASR_log, prefix="Epoch:[{}]".format(epoch))

    model.train()
    logging.info("Using {}".format(device))
    end = time.time()
    for i, (input, target, poisoned) in enumerate(train_loader):
        num_poisoned = torch.sum(poisoned)
        data_time.update(time.time() - end)
        end = time.time()

        input = Variable(input).to(device)
        target = Variable(target.long()).to(device)

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ASR_OUTPUT = []
        ASR = []
        PA_OUTPUT = []
        PA = []
        for idx in range(len(output)):
            if poisoned[idx]:
                ASR_OUTPUT.append(output[idx])
                ASR.append(target[idx])
            else:
                PA_OUTPUT.append(output[idx])
                PA.append(target[idx])

        asr = [[0.0]]
        if num_poisoned > 0:
            ASR_OUTPUT = torch.stack(ASR_OUTPUT).to(device)
            ASR = torch.Tensor(ASR).to(device)
            asr = accuracy(ASR_OUTPUT, ASR)
        PA_OUTPUT = torch.stack(PA_OUTPUT).to(device)
        PA = torch.Tensor(PA).to(device)
        pa = accuracy(PA_OUTPUT, PA)

        losses.update(loss.item(), input.size(0))
        ASR_log.update(asr[0][0], num_poisoned)
        PA_log.update(pa[0][0], PA_OUTPUT.size(0))

        batch_time.update(time.time() - end)

        if i % 100==0:
            progress.print(i)
        if scheduler is not None:
            scheduler.step()
            step += 1
    logging.info('====> PA {pa.avg:.3f} ASR {asr.avg:.3f}'.format(pa=PA_log, asr=ASR_log))

    return PA_log.avg, ASR_log.avg, step

# Validation function
def attack_validate(val_loader, model, criterion, **kwargs):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    ASR_log = AverageMeter('ASR', ':6.2f')
    PA_log = AverageMeter('PA', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, PA_log, ASR_log, prefix='Test: ')
    device = kwargs.get('device')

    model.eval()
    logging.info("Using {}".format(device))
    with torch.no_grad():
        end = time.time()
        for i, (input, target, poisoned) in enumerate(val_loader):
            num_poisoned = torch.sum(poisoned)
            end = time.time()

            input = Variable(input).to(device)
            target = Variable(target.long()).to(device)

            output = model(input)
            loss = criterion(output, target)

            ASR_OUTPUT = []
            ASR = []
            PA_OUTPUT = []
            PA = []
            for idx in range(len(output)):
                if poisoned[idx]:
                    ASR_OUTPUT.append(output[idx])
                    ASR.append(target[idx])
                else:
                    PA_OUTPUT.append(output[idx])
                    PA.append(target[idx])

            asr = [[0.0]]
            if num_poisoned > 0:
                ASR_OUTPUT = torch.stack(ASR_OUTPUT).to(device)
                ASR = torch.Tensor(ASR).to(device)
                asr = accuracy(ASR_OUTPUT, ASR)
            PA_OUTPUT = torch.stack(PA_OUTPUT).to(device)
            PA = torch.Tensor(PA).to(device)
            pa = accuracy(PA_OUTPUT, PA)

            losses.update(loss.item(), input.size(0))
            ASR_log.update(asr[0][0], num_poisoned)
            PA_log.update(pa[0][0], PA_OUTPUT.size(0))

            batch_time.update(time.time() - end)

            if i % 30 == 0:
                progress.print(i)
        logging.info('====> PA {pa.avg:.3f} ASR {asr.avg:.3f}'.format(pa=PA_log, asr=ASR_log))

    return PA_log.avg, ASR_log.avg, losses.avg