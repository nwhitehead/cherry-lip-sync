import torch
import torch.nn as nn

class ClassesLoss:
    def __init__(self, classes=None, weight=1.0):
        self.criterion = nn.CrossEntropyLoss()
        self.classes = classes
        self.weight = weight

    def __call__(self, left, right):
        #left_class = self.classes[left.to(torch.long)].to(torch.float)
        #right_class = self.classes[right.to(torch.long)].to(torch.float)

        return self.criterion(left, right)
