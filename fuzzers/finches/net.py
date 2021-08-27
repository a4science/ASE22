import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn._reduction as _Reduction

class Net(nn.Module):
    def __init__(self, in_size, out_size):
        super(Net, self).__init__()
        layer_size = [in_size, 4096, out_size]
        self.fc1 = nn.Linear(layer_size[0], layer_size[1])
        self.fc2 = nn.Linear(layer_size[1], layer_size[2])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class DistLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(DistLoss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        zero_loss_indices = torch.where(input == 1.0)
        target[zero_loss_indices] = 1.0
        return F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)

