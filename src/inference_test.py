import torch
from model import NeuralNet
from data import LipsyncDataset, AudioMFCC, Upsample, Downsample, PadVisemes, RandomChunk

# Audiorate
rate = 16000

viseme_labels = ['Ah', 'D', 'Ee', 'F', 'L', 'M', 'Neutral', 'Oh', 'R', 'S', 'Uh', 'Woo']

# Hyper-parameters
mels = 13
feature_dims = mels * 2
lookahead_frames = 6
input_size = feature_dims
hidden_size = 100
num_classes = len(viseme_labels)
num_epochs = 200
batch_size = 10
layers = 2

model = NeuralNet(input_size, hidden_size, layers, num_classes)

d = torch.load('checkpoints/model-199.pt', weights_only=True, map_location=torch.device('cpu'))
model.load_state_dict(d)
model.eval()

# No transform so we get raw audio and visemes for reference video generation
dataset = LipsyncDataset('./data/lipsync.parquet', transform=None)
s = dataset[5]
dataset.make_video(s['audio'], s['visemes'], filename='out_ref.mp4')
t = AudioMFCC(num_mels=mels)
ds = Downsample()
ma = t(s)['audio']
print(ma.shape)
ma = torch.unsqueeze(ma, 0)
print(ma.shape)
ma = ma.permute(0, 2, 1)
print(ma.shape)
out = model(ma)
print(out.shape)
_, v = torch.max(out.data, 2)
v = v[:, lookahead_frames:]
print(v.shape)
print(v)
dsv = ds({ 'audio': s['audio'], 'visemes': v, })['visemes']
print(dsv)
print(dsv.shape)
ref = s['visemes'][:dsv.shape[0]]
print(ref)
print(ref.shape)
print((dsv == ref).sum().item())
dataset.make_video(s['audio'], dsv.to(torch.long), filename='out_test.mp4')

