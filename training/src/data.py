
import tempfile
import numpy as np
import torch.nn as nn
import torch
import torchaudio
import torchvision
from torchvision.transforms import v2
from torch.utils.data import Dataset
import pyarrow.parquet as pq
from torch import Tensor
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
import torchaudio.functional as F

class CustomMFCC(torch.nn.Module):
    __constants__ = ["sample_rate", "n_mfcc"]

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        melkwargs = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.amplitude_to_DB = AmplitudeToDB("power", None)
        melkwargs = melkwargs or {}
        self.MelSpectrogram = MelSpectrogram(sample_rate=self.sample_rate, **melkwargs)

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: specgram_mel_db of size (..., ``n_mfcc``, time).
        """
        mel_specgram = self.MelSpectrogram(waveform)
        mel_specgram = self.amplitude_to_DB(mel_specgram)
        return mel_specgram

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
        a = torch.Tensor(self.table['audio'][idx].copy()).to(torch.float)
        v = torch.Tensor(self.table['visemes'][idx].copy()).to(torch.long)
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

    def make_video(self, audio, vis_indexes, filename='out.mp4'):
        """Make video given visemes (only for untransformed audio)"""
        sprites = v2.Resize((256 * 4, 256 * 4))(torchvision.io.decode_image('./images/demolipssheet_bg.png', 'RGB')).permute([1, 2, 0])
        frames = vis_indexes.shape[0]
        # Make copy of audio in stereo contiguous format for writing to video
        a = np.ascontiguousarray(audio.reshape(1, -1).expand(2, -1).numpy())
        v = torch.zeros(frames, 256, 256, 3)
        for i in range(frames):
            vi = self.viseme_labels[vis_indexes[i]]
            pos = self.visemes[vi]
            v[i, :, :, :] = sprites[pos[0] * 256 : pos[0] * 256 + 256, pos[1] * 256 : pos[1] * 256 + 256, :]
        #with tempfile.NamedTemporaryFile(delete_on_close=False, suffix='.mp4', dir='') as f:
        torchvision.io.write_video(filename, v, fps=30, audio_array=a, audio_fps=self.rate, audio_codec='aac')
        return filename

    def display_video(self, idx):
        sample = self[idx]
        fname = self.make_video(sample['audio'], sample['visemes'])
        return Video(fname)

class AudioMFCC(nn.Module):
    '''Analyze audio to MFCC'''
    # Times in seconds
    def __init__(self, audio_rate=16000, num_mels=13, window_time=25e-3, hop_time=10e-3):
        super().__init__()
        self.window_length = round(window_time * audio_rate)
        self.hop_length = round(hop_time * audio_rate)
        melkwargs = {
            "n_fft": self.window_length,
            "win_length": self.window_length,
            "hop_length": self.hop_length,
            "n_mels": num_mels,
            "center": False,
        }
        self.mfcc = CustomMFCC(sample_rate=audio_rate, n_mfcc=num_mels, melkwargs=melkwargs)

    def __call__(self, sample):
        waveform = sample['audio']
        a = self.mfcc(waveform)
        v = sample['visemes']
        # Convolve to get smoothed derivative at same size for everything
        d = np.array([0.5, 0.5, -0.5, -0.5])
        x = a.numpy()
        delta_a = np.apply_along_axis(np.convolve, axis=1, arr=x, v=d, mode='same')
        # Stack everything
        a = torch.cat((a, torch.tensor(delta_a)))
        return {
            'audio': a.to(torch.float),
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
        v = self.transform_viseme(sample['visemes'].reshape((1, 1, -1)).to(torch.float)).reshape((-1,))
        return {
            'audio': a,
            'visemes': v,
        }

class Downsample(nn.Module):
    '''Downsample visemes to new framerate (ensure no single frame durations)'''
    def __init__(self, old_fps=100, new_fps=30):
        super().__init__()
        ratio = new_fps / old_fps
        self.transform_viseme = nn.Upsample(scale_factor=ratio, mode='nearest-exact')

    def __call__(self, sample):
        a = sample['audio']
        # Visemes needs to have batch etc. stuff in front, then also be float to work
        v = self.transform_viseme(sample['visemes'].reshape((1, 1, -1)).to(torch.float)).reshape((-1,))
        vout = v[:]
        last_viseme = -1
        dur = 2
        for i in range(v.shape[0]):
            if dur > 1:
                if v[i] == last_viseme:
                    dur += 1
                else:
                    dur = 1
                    last_viseme = v[i]
            else:
                dur += 1
            vout[i] = last_viseme

        return {
            'audio': a,
            'visemes': vout,
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
            'audio': aa.to(torch.float),
            'visemes': vv.to(torch.long),
        }
