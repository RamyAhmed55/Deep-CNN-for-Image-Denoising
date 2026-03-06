from torch import nn

class DnCNN(nn.Module):
    """
    Non-BN version.
    can be trained to predict residual if the training targets are residual
    """
    def __init__(self, num_layers=20, num_features=64, in_ch=1, out_ch=1):
        super().__init__()
        layers = []
        layers += [nn.Conv2d(in_ch, num_features, 3, 1, 1, bias=True),
                   nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True),
                       nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(num_features, out_ch, 3, 1, 1, bias=True)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
def build_model(arch: str):
    if arch == "DnCNN-S":
        return DnCNN(num_layers=17)
    elif arch == "DnCNN-B":
        return DnCNN(num_layers=20)
    elif arch == "DnCNN-3":
        return DnCNN(num_layers=20)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    