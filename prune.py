import logging
import torch
import numpy as np
import process

def prune_by_ratio(weight, bias, ratio, device):
    num_weight = torch.numel(weight)
    
    # Weight mask
    threshold = np.sort(weight.abs().flatten())[int(num_weight*ratio)]
    weight_mask = torch.ge(weight.abs(), threshold).type('torch.FloatTensor').to(device)

    # Bias mask
    bias_mask = torch.ones(bias.size()).to(device)
    for i in range(bias_mask.size(0)):
        if len(torch.nonzero(weight_mask[i]).size()) == 0:
            bias_mask[i] = 0

    prune_ratio = (num_weight - torch.nonzero(weight_mask).size(0))/num_weight
    
    return weight_mask, bias_mask, prune_ratio


def prune_by_nueron(weight, bias, number, device, ascending=False):
    num_weight = torch.numel(weight)

    sign = 1 if ascending else -1
    
    # Weight mask
    threshold = sign*np.sort(sign*weight.flatten())[number]
    weight_mask = torch.lt(weight, threshold).type('torch.FloatTensor').to(device)

    # Bias mask
    bias_mask = torch.ones(bias.size()).to(device)
    for i in range(bias_mask.size(0)):
        if len(torch.nonzero(weight_mask[i]).size()) == 0:
            bias_mask[i] = 0

    prune_ratio = (num_weight - torch.nonzero(weight_mask).size(0))/num_weight
    
    return weight_mask, bias_mask, prune_ratio

def prune(model, criterion, loader, prune_layers, **kwargs):
    device = kwargs.get('device')
    acc = []
    for name, m in model.named_modules():
        if name in prune_layers:
            logging.info("Testing layer {}".format(name))
            prune_number = list(range(0, m.out_features, m.out_features//1000))
            for number in prune_number:
                logging.info('Pruning top %d neurons' % number)
                weight = m.weight.data.cpu()
                bias = m.bias.data.cpu()

                weight_mask, bias_mask, _ = prune_by_nueron(weight, bias, number, device)
                
                m.weight.data *= weight_mask
                m.bias.data *= bias_mask

                top1, _, _ = process.validate(loader, model, criterion, device=device)
                acc.append([number, top1])

                m.weight.data = weight.to(device)
                m.bias.data = bias.to(device)
    return acc