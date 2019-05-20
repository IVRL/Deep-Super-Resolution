import torch
import torch.nn as nn

EPS = 1e-3
        
class ResidualLearningNet(nn.Module):
    def __init__(self, wavelets=False, l_channel=False):
        super(ResidualLearningNet, self).__init__()
        
        channels = 3
        if l_channel:
            channels = 1
        if wavelets:
            channels *= 4
        
        self.input = nn.Conv2d(in_channels=channels, out_channels=64,
                kernel_size=3, stride=1, padding=1, bias=False)
        self.residual_layer = self.make_layer(10)
        
        self.output = nn.Conv2d(in_channels=64, out_channels=channels,
                kernel_size=3, stride=1, padding=1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                
    def make_layer(self, count):
        layers = []
        for _ in range(count):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64,
                kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x2):
        residual = x2
        out = self.input(x2)
        out = self.relu(out)
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out
                                                                      
