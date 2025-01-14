import torch
from collections import OrderedDict

s = torch.load('./model-1-12.pt', map_location=torch.device('cpu'))
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
            out[f'{k}.{['r', 'z', 'n'][i]}'] = x
    else:
        out[k] = s[k]

torch.save(out, './model-1-12.ptx')