"""

This is a script to parse and organize Mozilla Common Voice audio clips into training
data for lip sync.


Example usage:
    uv run extract.py --cvroot=~/Downloads/old/cv-corpus-21.0-delta-2025-03-14/ --output ~/Downloads/old/cv-corpus-21.0-delta-2025-03-14/out.wav

"""

import argparse
import csv
from pathlib import Path
import random
import numpy as np
import torch
import torchaudio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cvroot', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--duration', type=float, default=60.0)
    parser.add_argument('--samplerate', type=int, default=48000)
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

    random.shuffle(clips)
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
    torchaudio.save(args.output, torch.tensor(audio_out), format='wav', sample_rate=target_samplerate)
