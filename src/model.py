import torch.nn as nn

# Simple LSTM with dropout and projection to correct output size
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, layers, num_classes):
        super(NeuralNet, self).__init__()
        layers = [
            nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True, dropout=0.5, proj_size=num_classes),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


