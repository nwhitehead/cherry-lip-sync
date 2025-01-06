
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
from torch.utils.data import Dataset

class LipsyncDataset(Dataset):
    """Audio to animated lip viseme dataset"""
    def __init__(self, parquet_file, transform=None, samplerate=16000):
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
        sample = {
            'audio': self.table['audio'][idx],
            'visemes': self.table['visemes'][idx],
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sorted
viseme_labels = ['Ah', 'D', 'Ee', 'F', 'L', 'M', 'Neutral', 'Oh', 'R', 'S', 'Uh', 'Woo']

# Audiorate
rate = 16000

def demo():
    dataset = LipsyncDataset('./data/lipsync.parquet')
    print(len(dataset))
    for i, sample in enumerate(dataset):
        print(i, sample['audio'].shape)

if __name__ == '__main__':
    demo()
