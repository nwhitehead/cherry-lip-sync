"""

Convert from text viseme distribution output from ProcessWAV.exe to numpy binary format.

"""

import argparse
import numpy as np
import csv
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    with open(args.input, 'rt', newline='') as f:
        reader = csv.reader(f, delimiter=' ')
        data = np.array([[float(x) for x in row] for row in reader], dtype=float)
    np.save(args.output, data)
