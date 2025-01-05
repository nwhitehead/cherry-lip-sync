import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import argparse
import random
import torch
import torchaudio
import pyloudnorm as pyln

table = pq.read_table(os.path.join('data', 'TIMIT', 'train-00000-of-00002.parquet')).to_pandas()

rate = 16000
meter = pyln.Meter(rate)

def extract(n, level=-20.0):
    data = table['audio'][n]['array']
    # Normalize each sample before concat
    loudness = meter.integrated_loudness(data)
    return pyln.normalize.loudness(data, loudness, level)

def extracts(duration, seed=1234, gap=0.5, level=-20.0):
    silence = np.zeros((round(gap * rate), ))
    random.seed(seed)
    index = list(range(table.shape[0]))
    random.shuffle(index)
    res = np.zeros((0,))
    while res.shape[0] < duration * rate and len(index) > 0:
        i = index.pop()
        print(f'Extracting sentence {i}')
        res = np.concatenate([res, extract(i, level=level), silence])
    return res

def main():
    parser = argparse.ArgumentParser(description='Generate audio files with random sentences')
    parser.add_argument('--duration', type=float, default=5.0, metavar='SEC', help='how many seconds of audio to request (may be exceeded)')
    parser.add_argument('--seed', type=int, default=1234, help='random seed to use')
    parser.add_argument('--gap', type=float, default=0.5, metavar='SEC', help='how long between sentences')
    parser.add_argument('--level', type=float, default=-20.0, metavar='LUFS', help='perceptual volume level')
    parser.add_argument('--output', default='out.mp3', help='output file')
    args = parser.parse_args()
    print(args)
    waveform = extracts(duration=args.duration, seed=args.seed, gap=args.gap, level=args.level)
    total_duration = waveform.shape[0] / rate
    print(f'Total duration: {total_duration} s')
    print(f'Writing output: {args.output}')
    torchaudio.save(args.output, torch.Tensor(waveform.reshape([1, -1])), sample_rate=rate, backend='ffmpeg')

if __name__ == '__main__':
    main()
