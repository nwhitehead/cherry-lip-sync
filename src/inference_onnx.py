import torch
import torchaudio
import onnxruntime
import numpy as np

ort_session = onnxruntime.InferenceSession("./data/model-2-80-dropout.onnx")

from data import LipsyncDataset, AudioMFCC, Upsample, Downsample, PadVisemes, RandomChunk

# Audiorate
rate = 16000

sample, r = torchaudio.load('./data/male.wav')
sample = sample[0, :] # just get one channel
print(sample.shape)
sample = torchaudio.functional.resample(sample, r, rate)
print(sample.shape)

viseme_labels = ['Ah', 'D', 'Ee', 'F', 'L', 'M', 'Neutral', 'Oh', 'R', 'S', 'Uh', 'Woo']

# Hyper-parameters
mels = 13
feature_dims = mels * 2
lookahead_frames = 6
input_size = feature_dims
hidden_size = 80
num_classes = len(viseme_labels)
layers = 2

# No transform so we get raw audio and visemes for reference video generation
dataset = LipsyncDataset('./data/lipsync.parquet', transform=None)

s = {
    'audio': sample,
    'visemes': torch.tensor([]),
}

t = AudioMFCC(num_mels=mels)
ds = Downsample()
ma = t(s)['audio'].unsqueeze(0).permute(0, 2, 1).cpu().numpy()
print(ma.shape, ma[0, 20, :])
inputs =  {'input': ma}
print(inputs)
out = ort_session.run(None, inputs)
o = np.array(out[0])
print(o.shape)
print(o[:, 0, :])

_, v = torch.max(torch.tensor(o), axis=2)
#v = v[:, lookahead_frames:]
dsv = ds({ 'audio': s['audio'], 'visemes': v, })['visemes']
print(dsv.to(torch.long))
dataset.make_video(s['audio'], dsv.to(torch.long), filename=f'out.mp4')
