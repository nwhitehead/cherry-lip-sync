import torch

data = torch.load('checkpoints/model-199.pt', weights_only=True, map_location=torch.device('cpu'))
print(data)
