import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

mean = lambda x : sum(x)/len(x)

def ema(items, window_size=100):
    ema_list = []
    for i in range(1, len(items)+1):
        if i < window_size+1:
            ema_list.append(mean(items[:i]))
        else:
            ema_list.append(mean(items[i-window_size:i]))
    return ema_list

def main():
    dir = "./trigger_data_analysis"
    items = os.listdir(dir)
    items.sort()
    for item in tqdm(items):
        neuron, trigger, (activation, prediction) = torch.load(os.path.join(dir, item), map_location='cpu')
        activation_baseline = activation[0]
        activation = activation[1:]
        prediction_baseline = prediction[1]
        original_prediction = []
        target_prediction = []
        clean_sample_original_prediction = prediction[0][0]
        clean_sample_target_prediction = prediction[0][1]
        for p in prediction[2:]:
            original_prediction.append(p[0])
            target_prediction.append(p[1])
        
        assert len(original_prediction) == len(activation)
        
        
        window_size = 100
        fig, axs = plt.subplots(2,1, figsize=(25,10))
        axs[0].set_title("Activation")
        axs[0].plot(list(range(1,10001)), activation, '-r', alpha=0.2)
        axs[0].plot(list(range(1,10001)), ema(activation, window_size), '-r')
        axs[0].legend(['EMA(0)',f'EMA({window_size})'])

        axs[1].set_title("Target Label Prediction")
        # axs[1].plot(list(range(1,10001)), [clean_sample_original_prediction] * 10000, '--g', alpha=0.5)
        axs[1].plot(list(range(1,10001)), [clean_sample_target_prediction] * 10000, '--g', alpha=0.5)
        # axs[1].plot(list(range(1,10001)), [prediction_baseline[0]] * 10000, '--k', alpha=0.3)
        axs[1].plot(list(range(1,10001)), [prediction_baseline[1]] * 10000, '--k', alpha=0.5)
        # axs[1].plot(list(range(1,10001)), original_prediction, '--r', alpha=0.2)
        axs[1].plot(list(range(1,10001)), target_prediction, '-r', alpha=0.2)
        axs[1].plot(list(range(1,10001)), ema(target_prediction), '-r', alpha=0.8)
        axs[1].legend(['Clean','Initial','EMA(0)',f'EMA({window_size})'])
        plt.suptitle(item[:-3].replace('_', ' ').upper())
        plt.savefig(os.path.join("trigger_plot", item[:-3]+'.png'))

if __name__ == '__main__':
    main()