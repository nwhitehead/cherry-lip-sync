import torch
import torch.nn as nn

class SelectItem(nn.Module):
    def __init__(self, index):
        super().__init__()
        self._name = 'SelectItem'
        self.index = index

    def forward(self, inputs):
        return inputs[self.index]

class Permute(nn.Module):
    def __init__(self, dims) -> None:
        super().__init__()
        self._name = 'Permute'
        self.dims = dims

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.permute(*self.dims)
