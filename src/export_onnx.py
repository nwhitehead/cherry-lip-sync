import torch

from model import NeuralNet

viseme_labels = ['Ah', 'D', 'Ee', 'F', 'L', 'M', 'Neutral', 'Oh', 'R', 'S', 'Uh', 'Woo']

# Hyper-parameters
mels = 13
feature_dims = mels * 2
lookahead_frames = 6
input_size = feature_dims
hidden_size = 80
num_classes = len(viseme_labels)
layers = 2
max_time = 1

model = NeuralNet(input_size, hidden_size, layers, num_classes)

x = torch.randn(1, max_time, feature_dims)

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
