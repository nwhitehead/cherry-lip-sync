"""

This is a script to parse and organize Mozilla Common Voice audio clips into training
data for lip sync.


Example usage:

    uv run extract.py --cvroot=~/Downloads/old/cv-corpus-21.0-delta-2025-03-14/ \
        --output ~/Downloads/old/cv-corpus-21.0-delta-2025-03-14/out \
        --num=10 \
        --loop=60 \
        --duration=60

This will produce `out/out-N.parquet` for N from 0 to 9.

"""

import argparse
import csv
from pathlib import Path
import random
import numpy as np
import torch
import torchaudio
import wave
import os
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cvroot', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--duration', type=float, default=60.0)
    parser.add_argument('--samplerate', type=int, default=48000)
    parser.add_argument('--num', type=int, default=2)
    parser.add_argument('--loop', type=int, default=2)
    parser.add_argument('--command', type=str, default='wine ../../../win/ProcessWAV.exe --print-viseme-distribution')
    args = parser.parse_args()
    random.seed(args.seed)

    root = Path(args.cvroot).expanduser() / args.language
    other_filename = root / 'other.tsv'
    duration_filename = root / 'clip_durations.tsv'
    # Durations is { filename: duration }
    durations = {}
    with open(duration_filename, newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        # ignore header
        next(reader)
        for row in reader:
            durations[row[0]] = int(row[1])

    clips = []
    with open(other_filename, newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        # ignore header
        next(reader)
        clips = [row[1] for row in reader]

    # Use same shuffling for all n in num
    # pop off elements and keep them off so we don't duplicate between n
    random.shuffle(clips)
    for on in range(args.num):
        for n in range(args.loop):
            result = []
            time = 0
            audio_out = np.zeros([1, 0])
            while time < args.duration * 1000:
                a = clips.pop()
                result.append(a)
                time += durations[a]
                audio, samplerate = torchaudio.load(root / 'clips' / a)
                target_samplerate = args.samplerate
                sample = torchaudio.functional.resample(audio, samplerate, target_samplerate)
                audio_out = np.concatenate((audio_out, sample), axis=1)
            outpath = f'{args.output}-{on}-{n}.wav'
            cmdoutpath = f'{args.output}-{on}-{n}.out'
            torchaudio.save(outpath, torch.tensor(audio_out), format='wav', sample_rate=target_samplerate)
            # Now simplify WAV file header by reading/writing it with wave module
            with wave.open(outpath, 'rb') as fin:
                params = fin.getparams()
                n = fin.getnframes()
                data = fin.readframes(n)
            with wave.open(outpath, 'wb') as fout:
                fout.setparams(params)
                fout.writeframes(data)
            print(f'Wrote {outpath} ({time / 1000.0} s)')
            if args.command:
                os.system(f'{args.command} {outpath} > {cmdoutpath}')

        # Collect all WAV and output into one big table
        entries = []
        parquetpath = f'{args.output}-{on}.parquet'
        for n in range(args.loop):
            outpath = f'{args.output}-{on}-{n}.wav'
            cmdoutpath = f'{args.output}-{on}-{n}.out'
            audio, samplerate = torchaudio.load(outpath)
            audio = audio.numpy()[0]
            with open(cmdoutpath, 'rt', newline='') as f:
                reader = csv.reader(f, delimiter=' ')
                data = [[float(x) for x in row] for row in reader]
            entries.append({ 'audio': audio, 'visemes': data })
        data = pd.DataFrame(entries)
        data.to_parquet(parquetpath)
