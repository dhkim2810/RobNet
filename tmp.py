import os
import torch
import torch.nn as nn
import data
import utils
from model import VGG16_BN

from utils import *
import time
from torch.autograd import Variable

# Validation function
def validate(val_loader, model, cuda=False):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, top1, top5, prefix='Test: ')

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            end = time.time()
            if cuda:
                input = Variable(input).cuda()
                target = Variable(target).cuda()
            else:
                input = Variable(input)
                target = Variable(target)

            output = model(input)

            acc1, acc5 = accuracy(output, target, topk=(1,5))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            batch_time.update(time.time() - end)

            if i % 10 == 0:
                progress.print(i)
        logging.info('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def main():
    args = utils.get_argument()

    mModel = VGG16_BN()
    mModel.load_state_dict(torch.load('benign.pth.tar'))
    _, test_loader = data.load_data(args)

    device = 'cpu'
    top1, top5 = validate(test_loader, mModel)

if __name__=="__main__":
    main()