import torch
from model import NeuralNet

model = NeuralNet(3, 3, 1, 3)

# model is randomly initialized
torch.save(model.state_dict(), './rust_inference/model-random3.pt')

d = torch.load('./rust_inference/model-random3.pt', weights_only=True, map_location=torch.device('cpu'))
model.load_state_dict(d)
model.eval()

x = torch.randn([1, 2, 3])
out = model(x)
torch.save({ 'test': x}, f'test_in.pt')
torch.save({ 'test': out}, f'test_out.pt')
