import numpy as np
import torch
import torchaudio
import torchvision
from torchvision.transforms import v2
import csv

import argparse

visemes = {
    'D': [0, 1],
    'B': [0, 3],
    'I': [0, 2],
    'G': [1, 3],
    'H': [1, 1],
    'A': [2, 0],
    'X': [1, 0],
    'E': [1, 2],
    'K': [2, 3],
    'J': [2, 1],
    'C': [2, 2],
    'F': [0, 0],
}

def composite(target, source, position):
  pass

def main(args):
    timing = []
    with open(args.sync, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            timing += [[float(row[0]), row[1]]]
    vsize = round(timing[-1][0] * args.fps)
    varray = []
    for i in range(vsize + 1):
        t = i / args.fps + 0.002
        vout = None
        for tv, v in timing:
            if tv < t:
                vout = v
        if vout is None:
            continue
        varray.append(vout)
    print(''.join(varray))

    bg = v2.Resize((256, 256))(torchvision.io.decode_image(args.bg, 'RGB')).permute([1, 2, 0])
    sprites = v2.Resize((256 * 4 // 4, 256 * 4 // 4))(torchvision.io.decode_image(args.sprites, 'RGBA')).permute([1, 2, 0])
    audio, audio_samplerate = torchaudio.load(args.audio)
    frames = len(varray)
    # Make copy of audio in stereo contiguous format for writing to video
    a = np.ascontiguousarray(audio.reshape(1, -1).expand(2, -1).numpy())
    v = torch.zeros(frames, 256, 256, 3)
    xoff = 128 - 32
    yoff = 128 + 20
    for i in range(frames):
        vi = varray[i]
        pos = visemes[vi]
        alpha = sprites[pos[0] * 256 // 4 : pos[0] * 256 // 4 + 256 // 4, pos[1] * 256 // 4 : pos[1] * 256 // 4 + 256 // 4, 3].unsqueeze(2)
        print(alpha.shape)
        v[i, :, :, :] = bg
        v[i, yoff:yoff+64, xoff:xoff+64, :] *= (255.0 - alpha ) / 255.0
        v[i, yoff:yoff+64, xoff:xoff+64, :] += (alpha / 255.0) * (sprites[pos[0] * 256 // 4 : pos[0] * 256 // 4 + 256 // 4, pos[1] * 256 // 4 : pos[1] * 256 // 4 + 256 // 4, :3])
    print(v.shape)
    torchvision.io.write_video(args.output, v, fps=int(args.fps), audio_array=a, audio_fps=audio_samplerate, audio_codec='aac')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', required=True)
    parser.add_argument('--sync', required=True)
    parser.add_argument('--sprites', required=True)
    parser.add_argument('--bg', required=False)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fps', type=float, default=30.0)
    args = parser.parse_args()
    main(args)
