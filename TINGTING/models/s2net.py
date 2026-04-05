import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import RegionAlignedSNN, GraphVectorKuramoto

class TopDownPathway(nn.Module):
    """
    Implements Section 2.2.1: Top-Down Pathway of Neural Oscillatory Synchronization.
    
    This module handles:
    1. Latent projection of control signals (Eq. 1).
    2. Coupled vector Kuramoto dynamics (Eq. 2).
    """
    def __init__(self, in_dim, latent_dim, osc_dim, k_coupling, dt, device):
        super().__init__()
        self.in_dim = in_dim
        self.osc_dim = osc_dim
        
        # Eq. 1: Latent projection logic
        # Gamma(t) = f_proj(f_enc(X(t)))
        self.encoder = nn.Linear(latent_dim, in_dim) # Represents natural frequency mapping
        self.projection = nn.Sequential(
            nn.Linear(in_dim, latent_dim), 
            nn.LayerNorm(latent_dim), 
            nn.LeakyReLU(0.1),            
            nn.Dropout(0.1),              
        )

        # Eq. 2: Vector-valued Kuramoto dynamics
        # d_theta/dt = omega + K * sum(A * sin(theta_j - theta_i - alpha))
        self.kuramoto = GraphVectorKuramoto(
            N=in_dim, 
            D=osc_dim,      
            K=k_coupling, 
            dt=dt, 
            device=device
        )

    def forward(self, x_t, theta_prev, sc_matrix):
        """
        Args:
            x_t: Input signal at time t.
            theta_prev: Oscillator state at t-1.
            sc_matrix: Structural Connectivity (A_ij in Eq. 2).
        """
        # 1. Extract control signal (Eq. 1)
        # See snippet 1: "Latent projection of control signals"
        z_t = self.projection(x_t)
        gamma_t = self.encoder(z_t) # This acts as the forcing term or natural freq adjustment

        # 2. Evolve dynamics (Eq. 2)
        # See snippet 2: "Coupled vector Kuramoto dynamics..."
        theta_curr = self.kuramoto(theta_prev, gamma_t, A=sc_matrix)
        
        return theta_curr

class S2Net(nn.Module):
    """
    S²-Net: Synchronized Spiking Network
    
    Integrates:
    - Top-Down Pathway (Kuramoto Dynamics)
    - Interaction Mechanism (Phase-Delay Gating, Eq. 4-5)
    - Bottom-Up Pathway (Rhythm-Modulated SNN, Eq. 6)
    """
    def __init__(self, T, num_regions, num_classes, args, device="cuda"):
        super().__init__()
        self.T = T
        self.num_regions = num_regions
        self.osc_dim = 4  # Vector dimension D
        self.phase_delay_tau = 2 # Tau in Eq. 4
        
        # --- Top-Down Pathway ---
        self.top_down = TopDownPathway(
            in_dim=num_regions,
            latent_dim=args.hidden,
            osc_dim=self.osc_dim,
            k_coupling=args.k,
            dt=args.dt,
            device=device
        )
        
        # --- Interaction / Gating ---
        # Used for Eq. 4: g_i(t) calculation
        self.mask_proj = nn.Linear(num_regions, num_regions) 

        # --- Bottom-Up Pathway ---
        # See snippet 3: "Rhythm-modulated spiking neural network"
        self.bottom_up = RegionAlignedSNN(
            T=T, 
            num_regions=num_regions, 
            input_feat_dim=self.osc_dim, 
            num_classes=num_classes,
            low_n=args.low_n, 
            high_n=args.high_n, 
            branch=args.branch, 
            device=device
        )
        
        self.temporal_smooth = nn.Conv1d(num_classes, num_classes, kernel_size=5, padding=2)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, x, sc):
        B, T, N = x.shape
        x = x.to(self.device)
        sc = sc.to(self.device)

        # Initialize Phase State Theta (Eq. 2)
        theta = torch.zeros(B, self.num_regions, self.osc_dim, device=self.device)
        theta_hist = []
        
        feats_list = [] 
        gating_masks_list = [] 

        # === Temporal Loop ===
        for t in range(T):
            x_t = x[:, t, :] 
            
            # 1. Update Top-Down Dynamics
            theta = self.top_down(x_t, theta, sc)
            theta_hist.append(theta)

            # 2. Interaction & Gating Mechanism (Eq. 4 & Eq. 5)
            # Retrieve delayed phase state for binding (Tau delay)
            # See snippet 3: "time delay Tau serves as a critical bridge"
            idx = max(0, t - self.phase_delay_tau)
            theta_delayed = theta_hist[idx]
            
            # Calculate mean phase for gating intensity
            theta_mean = theta_delayed.mean(dim=-1) 
            
            # Eq. 4: g_i(t) = sigma_gate(sin(theta(t-tau)))
            # Here we approximate sigma_gate with 0.5*(1+sin) for soft gating [0,1]
            g_i_t = 0.5 * (1.0 + torch.sin(theta_mean)) 
            
            # Eq. 5: Rhythm signal gamma_tilde = g_i(t) * gamma_i(t)
            # Here gamma_i(t) is represented by the sine of the current phase
            phase_feat = torch.sin(theta) 
            rhythm_signal = phase_feat * g_i_t.unsqueeze(-1) # Element-wise modulation
            
            feats_list.append(rhythm_signal.unsqueeze(1))

            # Transform g_i(t) for SNN dendritic modulation
            mask_hidden = torch.sigmoid(self.mask_proj(g_i_t))
            gating_masks_list.append(mask_hidden)

        # === Bottom-Up Processing ===
        # Eq. 6: Modulated membrane potential update happens inside this block
        core_input = torch.cat(feats_list, dim=1)            
        all_gating_signals = torch.stack(gating_masks_list, dim=1) 

        # Forward pass through Spiking Network
        snn_out, spikes = self.bottom_up(core_input, gating_signals=all_gating_signals)
        
        # Spatiotemporal spiking feature aggregation (Snippet 3)
        smoothed_out = self.temporal_smooth(snn_out)
        
        return self.logsoftmax(smoothed_out), spikes