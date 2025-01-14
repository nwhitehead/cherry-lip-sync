import torch
from collections import OrderedDict
import sys

filename = sys.argv[1]

s = torch.load(filename, map_location=torch.device('cpu'), weights_only=True)
out = OrderedDict(s)
for k in s:
    if '_ih_' in k or '_hh_' in k:
        r = s[k].shape[0]
        if r % 3 != 0:
            raise Exception('Dimension size of layer not a multiple of 3')
        h = r // 3
        for i in range(3):
            if 'bias' in k:
                x = s[k][i * h:i * h + h]
            else:
                x = s[k][i * h:i * h + h, :]
            out[f'{k}.{["r", "z", "n"][i]}'] = x
    else:
        out[k] = s[k]

torch.save(out, f'{filename}x')