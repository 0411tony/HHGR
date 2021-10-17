import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn

class HGCN2(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGCN2, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, G):
        x = F.normalize(x)
        x = self.hgc1(x, G)
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.LongTensor):
        x = x.matmul(self.weight)
        x = x.long()
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        # x = torch.sparse.mm(G, x)
        return x
