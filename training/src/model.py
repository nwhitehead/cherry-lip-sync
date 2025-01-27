import torch.nn as nn
from ops import SelectItem, Permute

# Simple LSTM with dropout and projection to correct output size
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, layers, num_classes):
        super(NeuralNet, self).__init__()
        layers = [
            # Input is N T C_mfcc
            Permute((0, 2, 1)),
            # BatchNorm1d needs N C T
            nn.BatchNorm1d(input_size),
            Permute((0, 2, 1)),
            # Now we are back to N T C
            nn.Dropout(p=0.2),
        ] + [
            nn.GRU(input_size, hidden_size, batch_first=True),
            SelectItem(0),
        ] + [
            nn.Dropout(p=0.2),
            nn.GRU(hidden_size, hidden_size, batch_first=True),
            SelectItem(0),
            # GRU output is N T C_H
        ] * (layers - 1) + [
            nn.Linear(hidden_size, num_classes),
            # Now output is N T C_visemes)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
