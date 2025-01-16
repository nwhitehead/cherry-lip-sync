import torch.nn as nn
from ops import SelectItem

# Simple LSTM with dropout and projection to correct output size
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, layers, num_classes):
        super(NeuralNet, self).__init__()
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.Dropout(p=0.2),
        ] + [
            nn.GRU(hidden_size, hidden_size, batch_first=True),
            SelectItem(0),
        ] + [
            nn.Dropout(p=0.2),
            nn.GRU(hidden_size, hidden_size, batch_first=True),
            SelectItem(0),
        ] * (layers - 1) + [
            nn.Linear(hidden_size, num_classes),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
