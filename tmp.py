import torch
from model import VGG16_BN
def main():
    model = VGG16_BN()
    model.load_state_dict(torch.load("checkpoint/benign.pth.tar"))

    #### Target Probability Map
    return None

if __name__=="__main__":
    main()