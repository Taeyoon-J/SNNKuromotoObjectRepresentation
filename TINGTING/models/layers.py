import torch
import torch.nn as nn
from SNN.SNN_layers.vector_kuramoto import graphVectorKuramoto
from SNN.spiking_model import Rhy_spike_rnn_test_denri_wotanh_new, spike_dense_test_origin

class GraphVectorKuramoto(nn.Module):
    """Wrapper for the Vector Kuramoto implementation."""
    def __init__(self, N, D, K, dt, device):
        super().__init__()
        self.kuramoto = graphVectorKuramoto(
            N=N, D=D, K=K, dt=dt, alpha_scale=1.0, device=device
        )
    
    def forward(self, theta, gamma, A):
        return self.kuramoto(theta, gamma, A=A)

class RegionAlignedSNN(nn.Module):
    """
    Bottom-Up Pathway: Rhythm-Modulated Spiking Neural Network.
    Implements the dendritic integration and membrane potential updates described in Eq. 6.
    """
    def __init__(self, T, num_regions, input_feat_dim, num_classes, low_n, high_n, branch, device):
        super().__init__()
        self.input_adapter = nn.Linear(input_feat_dim, 1)
        
        # The core SNN layer that supports "external_masks" (gating signals)
        self.rnn_layer = Rhy_spike_rnn_test_denri_wotanh_new(
            input_dim=num_regions, 
            output_dim=num_regions, 
            tau_minitializer='uniform', low_m=0, high_m=4,
            tau_ninitializer='uniform', low_n=low_n, high_n=high_n, 
            vth=0.5, branch=branch, dt=1, device=device, bias=True
        )
        
        self.readout_layer = spike_dense_test_origin(
            input_dim=num_regions, 
            output_dim=num_classes,
            vth=0.5, dt=1, device=device, bias=True
        )
        self.device = device

    def forward(self, input_4d_seq, gating_signals):
        batch_size, seq_num, _, _ = input_4d_seq.shape
        self.rnn_layer.set_neuron_state(batch_size)
        self.readout_layer.set_neuron_state(batch_size)
        
        outputs = []
        spikes_hist = [] 

        for i in range(seq_num):
            # Input projection
            input_t = input_4d_seq[:, i, :, :] 
            currents = self.input_adapter(input_t).squeeze(-1) 

            # Rhythm modulation (gating) applied here
            # See Eq. 6: U_i(t) = (1-g_i(t))*U + g_i(t)*V
            g_t = gating_signals[:, i, :]
            
            _, spike_t = self.rnn_layer.forward(currents, g_t)
            spikes_hist.append(spike_t) 

            mem_readout, _ = self.readout_layer.forward(spike_t)
            outputs.append(mem_readout)
            
        outputs = torch.stack(outputs).permute(1, 2, 0) # [B, Classes, T]
        spikes = torch.stack(spikes_hist).permute(1, 2, 0) 

        return outputs, spikes