import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable
_EPS = 1e-10
from utils import get_offdiag_indices

class SimulationDecoder(nn.Module):
    """Simulation-based decoder."""

    def __init__(self, loc_max, loc_min, vel_max, vel_min, suffix):
        super(SimulationDecoder, self).__init__()

        self.loc_max = loc_max
        self.loc_min = loc_min
        self.vel_max = vel_max
        self.vel_min = vel_min

        self.interaction_type = suffix

        if '_springs' in self.interaction_type:
            print('Using spring simulation decoder.')
            self.interaction_strength = .1
            self.sample_freq = 1
            self._delta_T = 0.1
            self.box_size = 5.
        elif '_charged' in self.interaction_type:
            print('Using charged particle simulation decoder.')
            self.interaction_strength = 1.
            self.sample_freq = 100
            self._delta_T = 0.001
            self.box_size = 5.
        elif '_charged_short' in self.interaction_type:
            print('Using charged particle simulation decoder.')
            self.interaction_strength = .1
            self.sample_freq = 10
            self._delta_T = 0.001
            self.box_size = 1.
        else:
            print("Simulation type could not be inferred from suffix.")

        self.out = None

        # NOTE: For exact reproduction, choose sample_freq=100, delta_T=0.001

        self._max_F = 0.1 / self._delta_T

    def unnormalize(self, loc, vel):
        loc = 0.5 * (loc + 1) * (self.loc_max - self.loc_min) + self.loc_min
        vel = 0.5 * (vel + 1) * (self.vel_max - self.vel_min) + self.vel_min
        return loc, vel

    def renormalize(self, loc, vel):
        loc = 2 * (loc - self.loc_min) / (self.loc_max - self.loc_min) - 1
        vel = 2 * (vel - self.vel_min) / (self.vel_max - self.vel_min) - 1
        return loc, vel

    def clamp(self, loc, vel):
        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        vel[over] = -torch.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        vel[under] = torch.abs(vel[under])

        return loc, vel

    def set_diag_to_zero(self, x):
        """Hack to set diagonal of a tensor to zero."""
        mask = torch.diag(torch.ones(x.size(1))).unsqueeze(0).expand_as(x)
        inverse_mask = torch.ones(x.size(1), x.size(1)) - mask
        if x.is_cuda:
            inverse_mask = inverse_mask.cuda()
        inverse_mask = Variable(inverse_mask)
        return inverse_mask * x

    def set_diag_to_one(self, x):
        """Hack to set diagonal of a tensor to one."""
        mask = torch.diag(torch.ones(x.size(1))).unsqueeze(0).expand_as(x)
        inverse_mask = torch.ones(x.size(1), x.size(1)) - mask
        if x.is_cuda:
            mask, inverse_mask = mask.cuda(), inverse_mask.cuda()
        mask, inverse_mask = Variable(mask), Variable(inverse_mask)
        return mask + inverse_mask * x

    def pairwise_sq_dist(self, x):
        xx = torch.bmm(x, x.transpose(1, 2))
        rx = (x ** 2).sum(2).unsqueeze(-1).expand_as(xx)
        return torch.abs(rx.transpose(1, 2) + rx - 2 * xx)

    def forward(self, inputs, relations, rel_rec, rel_send, pred_steps=1):
        # Input has shape: [num_sims, num_things, num_timesteps, num_dims]
        # Relation mx shape: [num_sims, num_things*num_things]

        # Only keep single dimension of softmax output
        relations = relations[:, :, 1]

        loc = inputs[:, :, :-1, :2].contiguous()
        vel = inputs[:, :, :-1, 2:].contiguous()

        # Broadcasting/shape tricks for parallel processing of time steps
        loc = loc.permute(0, 2, 1, 3).contiguous()
        vel = vel.permute(0, 2, 1, 3).contiguous()
        loc = loc.view(inputs.size(0) * (inputs.size(2) - 1), inputs.size(1), 2)
        vel = vel.view(inputs.size(0) * (inputs.size(2) - 1), inputs.size(1), 2)

        loc, vel = self.unnormalize(loc, vel)

        offdiag_indices = get_offdiag_indices(inputs.size(1))
        edges = Variable(torch.zeros(relations.size(0), inputs.size(1) *
                                     inputs.size(1)))
        if inputs.is_cuda:
            edges = edges.cuda()
            offdiag_indices = offdiag_indices.cuda()

        edges[:, offdiag_indices] = relations.float()

        edges = edges.view(relations.size(0), inputs.size(1),
                           inputs.size(1))

        self.out = []

        for _ in range(0, self.sample_freq):
            x = loc[:, :, 0].unsqueeze(-1)
            y = loc[:, :, 1].unsqueeze(-1)

            xx = x.expand(x.size(0), x.size(1), x.size(1))
            yy = y.expand(y.size(0), y.size(1), y.size(1))
            dist_x = xx - xx.transpose(1, 2)
            dist_y = yy - yy.transpose(1, 2)

            if '_springs' in self.interaction_type:
                forces_size = -self.interaction_strength * edges
                pair_dist = torch.cat(
                    (dist_x.unsqueeze(-1), dist_y.unsqueeze(-1)),
                    -1)

                # Tricks for parallel processing of time steps
                pair_dist = pair_dist.view(inputs.size(0), (inputs.size(2) - 1),
                                           inputs.size(1), inputs.size(1), 2)
                forces = (
                        forces_size.unsqueeze(-1).unsqueeze(1) * pair_dist).sum(
                    3)
            else:  # charged particle sim
                e = (-1) * (edges * 2 - 1)
                forces_size = -self.interaction_strength * e

                l2_dist_power3 = torch.pow(self.pairwise_sq_dist(loc), 3. / 2.)
                l2_dist_power3 = self.set_diag_to_one(l2_dist_power3)

                l2_dist_power3 = l2_dist_power3.view(inputs.size(0),
                                                     (inputs.size(2) - 1),
                                                     inputs.size(1),
                                                     inputs.size(1))
                forces_size = forces_size.unsqueeze(1) / (l2_dist_power3 + _EPS)

                pair_dist = torch.cat(
                    (dist_x.unsqueeze(-1), dist_y.unsqueeze(-1)),
                    -1)
                pair_dist = pair_dist.view(inputs.size(0), (inputs.size(2) - 1),
                                           inputs.size(1), inputs.size(1), 2)
                forces = (forces_size.unsqueeze(-1) * pair_dist).sum(3)

            forces = forces.view(inputs.size(0) * (inputs.size(2) - 1),
                                 inputs.size(1), 2)

            if '_charged' in self.interaction_type:  # charged particle sim
                # Clip forces
                forces[forces > self._max_F] = self._max_F
                forces[forces < -self._max_F] = -self._max_F

            # Leapfrog integration step
            vel = vel + self._delta_T * forces
            loc = loc + self._delta_T * vel

            # Handle box boundaries
            loc, vel = self.clamp(loc, vel)

        loc, vel = self.renormalize(loc, vel)

        loc = loc.view(inputs.size(0), (inputs.size(2) - 1), inputs.size(1), 2)
        vel = vel.view(inputs.size(0), (inputs.size(2) - 1), inputs.size(1), 2)

        loc = loc.permute(0, 2, 1, 3)
        vel = vel.permute(0, 2, 1, 3)

        out = torch.cat((loc, vel), dim=-1)
        # Output has shape: [num_sims, num_things, num_timesteps-1, num_dims]

        return out
