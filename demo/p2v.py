import argparse
import json

# Viseme set I'm using is same as Adobe Animator (12 images for lip sync)
visemes = { 'X', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K' }

# Map goes from phoneme to list of visemes.

# Decision about what to put on right side was made by looking at pictures of phonemes and pictures of visemes and picking
# things that looked reasonably close.
# Also tried to be consistent with:
# https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=4f3656da4cbd1979a5fa763f90defa1f90d53583
# Page 15

# AX EL EN missing?

p2v_map = {
    '_': ['X'],
    ',': ['X'],

    # Lip rounding vowel classes
    # "ah"
    'AO': ['C'],
    'AH': ['C'],
    'AA': ['D'],
    'ER': ['E'],
    'OY': ['E', 'I'],
    'AW': ['C', 'F'],
    'HH': ['C'],

    'UW': ['F'],
    'UH': ['F'],
    'OW': ['E', 'F'],

    'AE': ['C'],
    'EH': ['C'],
    'EY': ['C', 'I'],
    'AY': ['C', 'I'],

    'IH': ['I'],
    'IY': ['I'],

    # Alveolar semivowels (viseme split on L here?)
    'L': ['H'],
    'R': ['K'],
    'Y': ['K'],

    # Alveolar fricatives
    'S': ['B'],
    'Z': ['B'],

    # Alveolar
    'T': ['B'],
    'D': ['B'],
    'N': ['B'],

    # Palato alveolar
    'SH': ['J'],
    'ZH': ['J'],
    'CH': ['J'],
    'JH': ['J'],

    # Bilabial
    'P': ['A'],
    'B': ['A'],
    'M': ['A'],

    # Dental (reusing viseme class)
    'TH': ['B'],
    'DH': ['B'],

    # Labio-dental
    'F': ['G'],
    'V': ['G'],

    # Velar (could be new viseme shape? split too?)
    'NG': ['B'],
    'K': ['B'],
    'G': ['F'],
    'W': ['F'],
}

def to_visemes(phonemes):
    """Convert list of phonemes with timestamps to list of visemes with timestamps (output length may change)"""
    # phoneme_list must include silence at end
    assert len(phonemes) > 0
    assert phonemes[-1][0] == ','
    phonemes.append(['X', phonemes[-1][1]])
    visemes = []
    vtimestamps = []
    for i, (p, t) in list(enumerate(phonemes[:-1])):
        ph = p.rstrip('012').upper()
        if ph in p2v_map:
            v = p2v_map[ph]
        else:
            v = ['X']
        visemes.extend(v)
        end_time = phonemes[i + 1][1]
        n = len(v)
        vtimestamps.extend([t + i / n * (end_time - t) for i in range(n)])
    return list(zip(visemes, vtimestamps))


def main(args):
    with open(args.input, 'rt') as fin:
        rawdata = fin.read()
        data = json.loads(rawdata)
        phonemes = data['phonemes']
    res = []
    for x in phonemes:
        phoneme = x['phoneme']
        time = x['time'] / 1000
        res.append([phoneme, time])
    res = to_visemes(res)
    res.insert(0, ['X', 0])
    txt = ''.join([f'{p[1]}\t{p[0]}\n' for p in res])
    with open(args.output, 'wt') as fout:
        fout.write(txt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input filename in JSON format with phoneme timing information')
    parser.add_argument('--output', required=True, help='Output file to write TSV viseme information')
    args = parser.parse_args()
    main(args)
