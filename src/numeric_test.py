import torch
from model import NeuralNet

# input, hidden, layers
model = NeuralNet(2, 3, 1, 1)

# model is randomly initialized
torch.save(model.state_dict(), './rust_inference/model-random23.pt')

d = torch.load('./rust_inference/model-random23.pt', weights_only=True, map_location=torch.device('cpu'))
model.load_state_dict(d)
model.eval()

x = torch.randn([1, 5, 2])
out = model(x)
torch.save({ 'test': x}, f'./data/test_in23.pt')
torch.save({ 'test': out}, f'./data/test_out23.pt')
