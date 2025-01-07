
# Lip shapes from:
# https://graphicmama.com/blog/free-mouth-shapes-character-animator-puppet/

import os
import argparse
import random

import numpy as np
import torch.nn as nn
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchaudio
from torch.utils.data import Dataset



# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)





class LipsyncDataset(Dataset):
    """Audio to animated lip viseme dataset"""

    viseme_labels = ['Ah', 'D', 'Ee', 'F', 'L', 'M', 'Neutral', 'Oh', 'R', 'S', 'Uh', 'Woo']
    
    visemes = {
        'Ah': [0, 1],
        'D': [0, 3],
        'Ee': [0, 2],
        'F': [1, 3],
        'L': [1, 1],
        'M': [2, 0],
        'Neutral': [1, 0],
        'Oh': [1, 2],
        'R': [2, 3],
        'S': [2, 1],
        'Uh': [2, 2],
        'Woo': [0, 0],
    }

    def __init__(self, parquet_file, transform=None, samplerate=16000, table=None):
        if table is not None:
            self.table = table
        else:
            self.table = pq.read_table(parquet_file).to_pandas()
        self.transform = transform
        self.rate = samplerate

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx < 0 or idx >= len(self):
            raise IndexError()
        a = torch.Tensor(self.table['audio'][idx].copy())
        v = torch.Tensor(self.table['visemes'][idx].copy())
        sample = {
            'audio': a,
            'visemes': v,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    def display_audio(self, idx):
        """Show audio output (only for untransformed audio)"""
        sample = self[idx]
        return Audio(sample['audio'], rate=self.rate)

    def make_video(self, audio, vis_indexes):
        """Make video given visemes (only for untransformed audio)"""
        sprites = v2.Resize((256 * 4, 256 * 4))(torchvision.io.decode_image('../images/demolipssheet_bg.png', 'RGB')).permute([1, 2, 0])
        frames = vis_indexes.shape[0]
        # Make copy of audio in stereo contiguous format for writing to video
        a = np.ascontiguousarray(torch.Tensor(audio.copy()).reshape(1, -1).expand(2, -1).numpy())
        v = torch.zeros(frames, 256, 256, 3)
        for i in range(frames):
            vi = self.viseme_labels[vis_indexes[i]]
            pos = self.visemes[vi]
            v[i, :, :, :] = sprites[pos[0] * 256 : pos[0] * 256 + 256, pos[1] * 256 : pos[1] * 256 + 256, :]
        with tempfile.NamedTemporaryFile(delete_on_close=False, suffix='.mp4', dir='') as f:
            torchvision.io.write_video('out.mp4', v, fps=30, audio_array=a, audio_fps=self.rate, audio_codec='aac')
            return 'out.mp4'

    def display_video(self, idx):
        sample = self[idx]
        fname = self.make_video(sample['audio'], sample['visemes'])
        return Video(fname)

class AudioMFCC(nn.Module):
    '''Analyze audio to MFCC'''
    def __init__(self, audio_rate=16000, num_mels=13):
        super().__init__()
        window_time = 25e-3 # seconds
        window_length = round(window_time * audio_rate)
        hop_time = 10e-3 # seconds
        hop_length = round(hop_time * audio_rate)
        melkwargs = {
            "n_fft": window_length,
            "win_length": window_length,
            "hop_length": hop_length,
        }
        self.transform_audio = torchaudio.transforms.MFCC(sample_rate=audio_rate, n_mfcc=num_mels, melkwargs=melkwargs)

    def __call__(self, sample):
        a = self.transform_audio(sample['audio'])
        v = sample['visemes']
        return {
            'audio': a,
            'visemes': v,
        }

class Upsample(nn.Module):
    '''Upsample visemes to new framerate'''
    def __init__(self, old_fps=30, new_fps=100):
        super().__init__()
        ratio = new_fps / old_fps
        self.transform_viseme = nn.Upsample(scale_factor=ratio, mode='nearest-exact')

    def __call__(self, sample):
        a = sample['audio']
        # Visemes needs to have batch etc. stuff in front, then also be float to work
        v = self.transform_viseme(sample['visemes'].reshape((1, 1, -1)).to(dtype=torch.float)).reshape((-1,))
        return {
            'audio': a,
            'visemes': v,
        }

class PadVisemes(nn.Module):
    '''Pad visemes by a frame if we need it to match audio size'''
    def __call__(self, sample):
        a = sample['audio']
        v = sample['visemes']
        if v.shape[-1] < a.shape[-1]:
            vv = torch.Tensor(v.shape[0] + 1)
            vv[:-1] = v[:]
            vv[-1] = v[-1]
            v = vv
        return {
            'audio': a,
            'visemes': v,
        }

class RandomChunk(nn.Module):
    '''Extract fixed size block from random position'''
    def __init__(self, size=100, seed=1234):
        super().__init__()
        self.size = size
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample):
        # If sample is too small, play it again
        a = sample['audio']
        v = sample['visemes']
        if a.shape[-1] < self.size:
            aa = torch.zeros(a.shape[0], self.size)
            # v[-1] should be neutral viseme
            vv = torch.ones(self.size) * v[-1]
            offset = self.rng.integers(0, self.size - a.shape[-1])
            aa[:, offset:offset + a.shape[1]] = a[:, :]            
            vv[offset:offset + v.shape[0]] = v[:]
        else:
            offset = self.rng.integers(0, a.shape[-1] - self.size)
            aa = a[:, offset:offset + self.size]
            vv = v[offset:offset + self.size]
        return {
            'audio': aa,
            'visemes': vv,
        }

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sorted
viseme_labels = ['Ah', 'D', 'Ee', 'F', 'L', 'M', 'Neutral', 'Oh', 'R', 'S', 'Uh', 'Woo']

# Audiorate
rate = 16000

# Hyper-parameters
feature_dims = 28
lookahead_frames = 6
input_size = feature_dims * lookahead_frames
hidden_size = 500
num_classes = len(viseme_labels)
num_epochs = 5
batch_size = 10
learning_rate = 0.001

model = NeuralNet(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
transform = nn.Sequential(
    Upsample(),
    AudioMFCC(),
    PadVisemes(),
    RandomChunk(size=200, seed=1),
)
dataset = LipsyncDataset('./data/lipsync.parquet', transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

total_step = len(dataset)
for epoch in range(num_epochs):
    for i, sample in enumerate(train_loader):
        # Move tensors to the configured device
        print(epoch, i, sample['audio'].shape, sample['visemes'].shape)
        audio = sample['audio'].to(device)
        visemes = sample['visemes'].to(device)
#        images = images.reshape(-1, 28*28).to(device)
#         labels = labels.to(device)
        
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if (i+1) % 100 == 0:
#             print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
#                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# def demo():
#     dataset = LipsyncDataset('./data/lipsync.parquet')
#     print(len(dataset))
#     for i, sample in enumerate(dataset):
#         print(i, sample['audio'].shape, sample['visemes'].shape)

# if __name__ == '__main__':
#     demo()
