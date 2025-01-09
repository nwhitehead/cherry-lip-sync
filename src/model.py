import torch.nn as nn

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        layers = [
            # nn.Linear(input_size, hidden_size),
            # nn.Dropout(),
            # nn.ReLU(),
            # nn.Linear(hidden_size, num_classes),
            # nn.Dropout(),
            nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.5, proj_size=num_classes),
            # nn.Softmax(dim=0),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


