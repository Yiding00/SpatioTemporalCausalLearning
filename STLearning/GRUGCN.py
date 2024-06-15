import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUGCN(nn.Module):
    '''Using GRU+GCN to generate hidden representation of spatio-temporal data'''

    def __init__(self, graph_kernel, n_in, n_hid, n_out=1, do_prob=0.):
        super(GRUGCN, self).__init__()
        self.gru = nn.GRU(n_in, n_hid, batch_first=True)
        self.mlp = MLP(n_hid, 2 * n_hid, n_hid, do_prob)
        self.fc = nn.Linear(n_hid, 4*n_out)
        # 111
        # self.fc = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_hid)
        self.graph_kernel = nn.Parameter(graph_kernel.clone().detach())

    def batch_norm(self, inputs):
        x = inputs.reshape(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.reshape(inputs.size(0), inputs.size(1), -1)

    def graph_conv(self, inputs):
        x = inputs.reshape(inputs.size(0), inputs.size(1), -1)
        A = torch.repeat_interleave(self.graph_kernel.unsqueeze(0), inputs.size(0), dim=0)
        x = torch.bmm(A, x)
        return x.reshape(inputs.size(0), -1, inputs.size(3))

    def forward(self, inputs):
        # inputs: [num_sample, num_node, num_timepoint, num_feature]
        x1 = inputs.reshape(-1, inputs.size(2), inputs.size(3))
        x, _ = self.gru(x1)
        x = self.batch_norm(x)
        x = x.reshape(inputs.size(0), inputs.size(1), inputs.size(2), -1)
        x = self.graph_conv(x)
        x1 = self.mlp(x)
        x2 = self.fc(x1)
        x3 = x2.reshape(inputs.size(0), inputs.size(1), inputs.size(2), -1)
        return x3
