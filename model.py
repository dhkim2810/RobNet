import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features

        self.num_fc = 2
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.classifier = nn.Linear(4096, 10)

        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(True)

         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, get_activation=None, neuron=None):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if get_activation == -1: # Get every FC layer output
            x = self.relu(self.fc1(x))
            fc1 = x
            x = self.relu(self.fc2(x))
            fc2 = x
            x = self.classifier(x)
            return x, [fc1, fc2]
        if get_activation in [1,2]:
            x = self.relu(self.fc1(x))
            if get_activation == 2:
                x = self.relu(self.fc2(x))
            if neuron is None:
                return x
            return x[:, neuron]
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def VGG16_BN():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    return VGG(make_layers(cfg, batch_norm=True))