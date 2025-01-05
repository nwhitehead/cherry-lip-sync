import argparse
import re
import numpy as np
import pandas as pd

frame_pattern = re.compile(r'Keyframe index="(\d+)" label="([^"]+)"')
justframe_pattern = re.compile(r'Keyframe')

def process(txt):
    m = frame_pattern.search(txt)
    if m:
        g = m.groups()
        index = int(g[0])
        label = g[1]
        return index, label
    else:
        if justframe_pattern.search(txt):
            raise 'Keyframe missed!'
    return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    with open(args.input, 'rt') as fin:
        points = []
        for txt in fin:
            index, label = process(txt)
            if index is not None:
                points.append((index, label))
        df = pd.DataFrame({
            'frame': np.array([pnt[0] for pnt in points], dtype='int32'),
            'label': pd.Categorical([pnt[1] for pnt in points]),
        })
        df.to_parquet(args.output)

if __name__ == '__main__':
    main()
