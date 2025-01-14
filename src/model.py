import torch.nn as nn
from ops import SelectItem

# Simple LSTM with dropout and projection to correct output size
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, layers, num_classes):
        super(NeuralNet, self).__init__()
        layers = [
            # nn.Dropout(p=0.2),
            nn.GRU(input_size, hidden_size, num_layers=layers, batch_first=True),
            SelectItem(0),
            # nn.Linear(hidden_size, num_classes),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
