# rhythmic_snn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from skipDHSRNN.SNN_layers.spike_neuron import *
from skipDHSRNN.SNN_layers.spike_dense import *


class RhythmDHSRNN(nn.Module):
    """
    SNN whose membrane update is gated by Kuramoto-derived rhythmic mask.
    mask_batch: [B, output_dim] for each time t.
    """

    def __init__(self, input_dim, output_dim, branch=4, device='cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.branch = branch

        # Dense dendritic layer
        self.pad = ((input_dim + output_dim) // branch * branch + branch -
                    (input_dim + output_dim)) % branch

        self.dense = nn.Linear(input_dim + output_dim + self.pad,
                               output_dim * branch)

        # τ parameters
        self.tau_m = nn.Parameter(torch.randn(output_dim))
        self.tau_n = nn.Parameter(torch.randn(output_dim, branch))

        # internal states
        self.mem = None
        self.spike = None
        self.d_input = None
        self.v_th = None

        self.create_mask()

    def create_mask(self):
        input_size = self.input_dim + self.output_dim + self.pad
        self.mask = torch.zeros(self.output_dim * self.branch, input_size).to(self.device)
        for i in range(self.output_dim):
            seq = torch.randperm(input_size)
            for j in range(self.branch):
                self.mask[i * self.branch + j,
                          seq[j * input_size // self.branch:(j + 1) * input_size // self.branch]] = 1

    def apply_mask(self):
        self.dense.weight.data = self.dense.weight.data * self.mask

    def set_neuron_state(self, B):
        self.mem = torch.zeros(B, self.output_dim).to(self.device)
        self.spike = torch.zeros(B, self.output_dim).to(self.device)
        self.d_input = torch.zeros(B, self.output_dim, self.branch).to(self.device)
        self.v_th = torch.ones(B, self.output_dim).to(self.device)

    def forward(self, x_t, rhythmic_mask):
        """
        x_t: input at time t: [B, input_dim]
        rhythmic_mask: [B, output_dim] from Kuramoto phases
        """

        B = x_t.size(0)
        beta = torch.sigmoid(self.tau_n)

        padding = torch.zeros(B, self.pad).to(self.device)
        k_input = torch.cat((x_t.float(), self.spike, padding), dim=-1)

        self.d_input = beta * self.d_input + (1 - beta) * \
                       self.dense(k_input).view(B, self.output_dim, self.branch)

        dend = torch.sum(self.d_input, dim=2)  # [B, output_dim]

        # batch-wise rhythmic gating
        # gated = dend * rhythmic_mask

        # membrane update
        # self.mem, self.spike = mem_update_pra(
        #     gated, self.mem, self.spike, self.v_th, self.tau_m, dt=1, device=self.device
        # )
        effective_input = dend * rhythmic_mask
        effective_leak  = self.tau_m / (1e-3 + rhythmic_mask)   # leak 
        self.mem, self.spike = mem_update_pra(
    effective_input,
    self.mem,
    self.spike,
    self.v_th,
    effective_leak,
    dt=1,
    device=self.device
)
        return self.spike
