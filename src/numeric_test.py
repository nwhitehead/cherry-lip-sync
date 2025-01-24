import torch
from model import NeuralNet

# input, hidden, layers
model = NeuralNet(26, 80, 2, 12)

# # model is randomly initialized
# torch.save(model.state_dict(), './rust_inference/model.pt')

d = torch.load('./rust/model/model.pt', weights_only=True, map_location=torch.device('cpu'))
model.load_state_dict(d)
model.eval()

x = torch.randn([1, 5, 26])
out = model(x)
torch.save({ 'test': x}, f'./data/test_in.pt')
torch.save({ 'test': out}, f'./data/test_out.pt')
