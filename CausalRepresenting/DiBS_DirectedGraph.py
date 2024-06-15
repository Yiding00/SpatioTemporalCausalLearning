import torch
import torch.nn as nn
import torch.nn.functional as F

# DiBS Eq.6, A generalization of VGAE

class DiBS_DirectedGraph(nn.Module):
    def forward(self, inputs):
        u_mean = inputs[:,:,:,0].unsqueeze(3)
        u_var = inputs[:,:,:,1].unsqueeze(3)
        v_mean = inputs[:,:,:,2].unsqueeze(3)
        v_var = inputs[:,:,:,3].unsqueeze(3)
        u_temp = torch.normal(mean=0, std=1, size=u_mean.size()).cuda()
        v_temp = torch.normal(mean=0, std=1, size=u_mean.size()).cuda()
        u = u_mean + u_temp*u_var
        v = v_mean + v_temp*v_var
        temp = u*v
        out = torch.sigmoid(temp)
        # return x
        return u, v, out
