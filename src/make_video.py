#import torchvision
import argparse

visemes = {
    'Ah': [0, 1],
    'D': [0, 3],
    'Ee': [0, 2],
    'F': [1, 3],
    'L': [1, 1],
    'M': [2, 0],
    'Neutral': [1, 0],
    'Oh': [1, 2],
    'R': [2, 3],
    'S': [2, 1],
    'Uh': [2, 2],
    'Woo': [0, 0],
}

def main(args):
    print(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', required=True)
    parser.add_argument('--sync', required=True)
    parser.add_argument('--sprites', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args)
