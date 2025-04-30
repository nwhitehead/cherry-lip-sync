"""

Simplify a WAV file to have short header suitable for broken WAV utilities.

"""

import argparse
import wave

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    with wave.open(args.input, 'rb') as fin:
        params = fin.getparams()
        n = fin.getnframes()
        data = fin.readframes(n)
        with wave.open(args.output, 'wb') as fout:
            fout.setparams(params)
            fout.writeframes(data)
