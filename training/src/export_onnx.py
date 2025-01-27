import torch

from model import NeuralNet
import onnxruntime
import numpy as np

viseme_labels = ['Ah', 'D', 'Ee', 'F', 'L', 'M', 'Neutral', 'Oh', 'R', 'S', 'Uh', 'Woo']

# Hyper-parameters
mels = 13
feature_dims = mels * 2
lookahead_frames = 3
input_size = feature_dims
hidden_size = 80
num_classes = len(viseme_labels)
layers = 2
max_time = 10

model = NeuralNet(input_size, hidden_size, layers, num_classes)

d = torch.load('./rust_inference/model/model.pt', weights_only=True, map_location=torch.device('cpu'))
model.load_state_dict(d)
model.eval()

x = torch.randn(1, max_time, feature_dims)
torch_out = model(x)

torch.onnx.export(
    model,
    x,
    'model.onnx',
    export_params=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': { 1: 'time' },
        'output': { 1: 'time'},
    }
)

ort_session = onnxruntime.InferenceSession("model.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
print(ort_inputs)
print(ort_inputs['input'].shape)
ort_outs = ort_session.run(None, ort_inputs)
out = np.array(ort_outs[0])

np.testing.assert_allclose(to_numpy(torch_out), out, rtol=1e-03, atol=1e-05)

print(out[0, 8, :])
print(torch_out[0, 8, :])
