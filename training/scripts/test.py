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
import glob

# TIMIT
table = pq.read_table(os.path.join('data', 'TIMIT', 'train-00000-of-00002.parquet')).to_pandas()

# LibriSpeech
paths = glob.glob('../**/*.flac', recursive=True)

rate = 16000
meter = pyln.Meter(rate)

def extract_timit(n):
    data = table['audio'][n]['array']
    return data

def extract_lj(path):
    data, rate = torchaudio.load(path)
    data = data.numpy()[0]
    return data

def extracts(duration, seed=1234, gap=0.5, level=-20.0, include_timit=True, include_libri=True):
    silence = np.zeros((round(gap * rate), ))
    timit_index = [('TIMIT', i) for i in range(table.shape[0])]
    lj_index = [('LJ', path) for path in sorted(paths)]
    random.seed(seed)
    index = []
    if include_timit:
        index.extend(timit_index)
    if include_libri:
        index.extend(lj_index)
    random.shuffle(index)
    res = np.zeros((0,))
    while res.shape[0] < duration * rate and len(index) > 0:
        entry = index.pop()
        kind = entry[0]
        i = entry[1]
        if kind == 'TIMIT':
            print(f'Extracting TIMIT sentence {i}')
            samples = extract_timit(i)
        if kind == 'LJ':
            print(f'Extracting LJ sentence {i}')
            samples = extract_lj(i)
        # Normalize each sample before concat
        loudness = meter.integrated_loudness(samples)
        samples = pyln.normalize.loudness(samples, loudness, level)

        res = np.concatenate([res, samples, silence])
    return res

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='Generate audio files with random sentences')
    parser.add_argument('--duration', type=float, default=5.0, metavar='SEC', help='how many seconds of audio to request (may be exceeded)')
    parser.add_argument('--seed', type=int, default=1234, help='random seed to use')
    parser.add_argument('--gap', type=float, default=0.5, metavar='SEC', help='how long between sentences')
    parser.add_argument('--level', type=float, default=-20.0, metavar='LUFS', help='perceptual volume level')
    parser.add_argument('--output', default='out.mp3', help='output file')
    parser.add_argument('--timit', type=str2bool, nargs='?', const=True, default=True, help='whether to include TIMIT dataset')
    parser.add_argument('--libri', type=str2bool, nargs='?', const=True, default=True, help='whether to include LibriSpeech dataset')
    args = parser.parse_args()
    print(args)
    waveform = extracts(duration=args.duration, seed=args.seed, gap=args.gap, level=args.level, include_timit=args.timit, include_libri=args.libri)
    total_duration = waveform.shape[0] / rate
    print(f'Total duration: {total_duration} s')
    print(f'Writing output: {args.output}')
    torchaudio.save(args.output, torch.Tensor(waveform.reshape([1, -1])), sample_rate=rate, backend='ffmpeg')

if __name__ == '__main__':
    main()
