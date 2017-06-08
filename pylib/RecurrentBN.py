import torch.nn as nn

class RBN2d(nn.Module):
    def __init__(self, ch):
        super(RBN2d, self).__init__()
        self.first_iter = True
        self.bn1 = nn.BatchNorm2d(ch)
        self.bn2 = nn.BatchNorm2d(ch)
        self.bn1.weight = self.bn2.weight
        self.bn1.bias = self.bn2.bias
        # self.bn_list = []
        # for i in range(0, iter_num):
        #     self.bn_list.append(nn.BatchNorm2d(ch))
    def forward(self, x):
        if self.first_iter:
            out = self.bn1(x)
            self.first_iter = False
        else:
            out = self.bn2(x)
            self.first_iter = True
        return out
