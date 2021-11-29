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

    outputs = []
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            end = time.time()
            input = Variable(input).to(device)
            target = Variable(target).to(device)

            output = model(input)
            loss = criterion(output, target)
            outputs.append(output)

            acc1, acc5 = accuracy(output, target, topk=(1,5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            batch_time.update(time.time() - end)

            if i % 30 == 0:
                progress.print(i)
        logging.info('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg, torch.cat(outputs, dim=0).cpu()


# Train function
def attack_train(train_loader, **kwargs):
    epoch = kwargs.get('epoch')
    model = kwargs.get('model')
    criterion = kwargs.get('criterion')
    optimizer = kwargs.get('optimizer')
    scheduler = kwargs.get('scheduler')
    device = kwargs.get('device')

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    ASR_log = AverageMeter('ASR', ':6.2f')
    PA_log = AverageMeter('PA', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, PA_log, ASR_log, prefix="Epoch:[{}]".format(epoch))

    model.train()
    end = time.time()
    outputs = []
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
        
        outputs.append(output)

        asr = pa = [[0.0]]
        if num_poisoned > 0:
            asr_output = output[poisoned.nonzero()].view(num_poisoned, -1)
            asr_target = target[poisoned.nonzero()].view(num_poisoned)
            # if len(asr_target.shape) == 0:
            #     asr_target = torch.Tensor([asr_target.long()], device='cpu')
            asr = accuracy(asr_output, asr_target)
        if input.shape[0]-num_poisoned > 0:
            pa_output = output[(poisoned==0).nonzero()].view(input.shape[0]-num_poisoned, -1)
            pa_target = target[(poisoned==0).nonzero()].view(input.shape[0]-num_poisoned)
            pa = accuracy(pa_output, pa_target)
        
        losses.update(loss.item(), input.size(0))
        ASR_log.update(asr[0][0], num_poisoned)
        PA_log.update(pa[0][0], len(pa_target))
        batch_time.update(time.time() - end)

        if i % (len(train_loader)//4) == 0:
            progress.print(i)
        if scheduler is not None:
            scheduler.step()
            # step += 1
    logging.info('====> PA {pa.avg:.3f} ASR {asr.avg:.3f} Loss {loss.avg:.4f}'.format(pa=PA_log, asr=ASR_log,loss=losses))
    
    return PA_log.avg, ASR_log.avg, losses.avg, torch.cat(outputs, dim=0).cpu()

# Validation function
def attack_validate(val_loader, model, criterion, **kwargs):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    ASR_log = AverageMeter('ASR', ':6.2f')
    PA_log = AverageMeter('PA', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, PA_log, ASR_log, prefix='Test: ')
    device = kwargs.get('device')

    outputs = []
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
            outputs.append(output)

            asr = pa = [[0.0]]
            if num_poisoned > 0:
                asr_output = output[poisoned.nonzero()].view(num_poisoned, -1)
                asr_target = target[poisoned.nonzero()].view(num_poisoned)
                # if len(asr_target.shape) == 0:
                #     asr_target = torch.Tensor([asr_target.long()], device='cpu')
                asr = accuracy(asr_output, asr_target)
            if input.shape[0]-num_poisoned > 0:
                pa_output = output[(poisoned==0).nonzero()].view(input.shape[0]-num_poisoned, -1)
                pa_target = target[(poisoned==0).nonzero()].view(input.shape[0]-num_poisoned)
                pa = accuracy(pa_output, pa_target)

            losses.update(loss.item(), input.size(0))
            ASR_log.update(asr[0][0], num_poisoned)
            PA_log.update(pa[0][0], len(pa_target))

            batch_time.update(time.time() - end)

            if i % (len(val_loader)//4) == 0:
                progress.print(i)
        logging.info('====> PA {pa.avg:.3f} ASR {asr.avg:.3f} Loss {loss.avg:.4f}'.format(pa=PA_log, asr=ASR_log,loss=losses))

    return PA_log.avg, ASR_log.avg, losses.avg