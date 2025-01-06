import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import argparse
import random
import torchaudio
import pyloudnorm as pyln
import glob
from IPython.display import Audio

fps = 30

# Sorted
viseme_labels = ['Ah', 'D', 'Ee', 'F', 'L', 'M', 'Neutral', 'Oh', 'R', 'S', 'Uh', 'Woo']

def find_splits(table, min_frames=30, min_pause_frames=20):
    last_i = 0
    last_split = 0
    last_dur = 0
    max_i = table.shape[0]
    splits = []
    for i in range(max_i):
        if table['label'][i] == 'Neutral':
            fi = int(table['frame'][i])
            fi2 = int(table['frame'][i + 1]) if i < max_i - 1 else 0
            dur = fi2 - fi
            if i > 0 and i < max_i - 1 and dur > min_pause_frames and fi - last_split >= min_frames:
                splits.append([last_split + last_dur // 2, fi + dur // 2])
                data = table[last_split:i]
                last_i = i
                last_split = fi
                last_dur = fi2 - fi
    return splits

def extract_samples(samples, rate, start, end, padding=10):
    start_sample = round(start / fps * rate)
    end_sample = round(end / fps * rate)
    padding_len = round(padding / fps * rate)
    pad = np.zeros((padding_len,), dtype='float32')
    res = np.concatenate([pad, samples[start_sample:end_sample], pad])
    desired_len = round((end - start + 2 * padding) / fps * rate)
    extra_padding = desired_len - res.shape[0]
    if extra_padding > 0:
        res = np.concatenate([res, np.zeros((extra_padding,), dtype='float32')])
    return res

def extract_visemes(table, start, end, padding=10):
    symbols = table[(table['frame'] >= start) & (table['frame'] < end)].copy()
    symbols['frame'] -= start
    symbols['frame'] += padding
    return pd.concat([pd.DataFrame([{ 'frame': 0, 'label': 'Neutral' }]), symbols], ignore_index=True)

def extract_all(table, samples, rate, padding=10):
    entries = []
    neutral = viseme_labels.index('Neutral')
    for start, end in find_splits(table):
        total_len = end - start + padding * 2
        samps = extract_samples(samples, rate, start, end, padding=padding)
        visemes = extract_visemes(table, start, end, padding=padding)
        viseme_frames = visemes['frame'].values.tolist() + [total_len]
        viseme_numbers = [viseme_labels.index(x) for x in visemes['label'].values] + [neutral]
        frames = []
        last_frame = 0
        last_num = -1
        for frame, num in zip(viseme_frames, viseme_numbers):
            frames.extend([int(last_num)] * (frame - last_frame))
            last_frame = frame
            last_num = num
        # print(list(zip(viseme_frames, viseme_numbers)))
        # print('F', start, end, len(frames), frames)
        assert frames[0] == neutral
        assert frames[-1] == neutral
        assert len(frames) == total_len
        #print(start, end, samples.shape[0], round(total_len / fps * rate))
        assert samps.shape[0] == round(total_len / fps * rate)
        entries.append({ 'audio': samps, 'visemes': frames })
        # print(entries[-1])
    return pd.DataFrame(entries)

def build():
    results = []
    for n in range(1, 7):
        table = pq.read_table(os.path.join('.', 'data', f'out-{n}-600.parquet')).to_pandas()
        samples, rate = torchaudio.load(f'./data/out-{n}-600.mp3')
        samples = samples.numpy()[0]
        data = extract_all(table, samples, rate)
        results.append(data)
    return pd.concat(results)

data = build()
print(data)
data.to_parquet('./data/lipsync.parquet')

