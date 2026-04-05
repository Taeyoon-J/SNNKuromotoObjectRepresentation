import torch
import torch.nn as nn
from .layers import GraphVectorKuramoto, RegionAlignedSNN

class S2NetClassifier(nn.Module):
    """
    S²-Net for Subject Classification.
    
    Differences from Sequence Labeling Model:
    1. Includes a temporal pooling layer (Global Average Pooling) at the end.
    2. Output shape is [Batch, NumClasses] instead of [Batch, NumClasses, Time].
    """
    def __init__(self, T, num_regions, num_classes, args, device="cuda"):
        super().__init__()
        self.T = T
        self.in_dim = int(num_regions)
        self.osc_dim = 4
        self.phase_delay_steps = 2
        self.latent_dim = args.hidden

        # --- Top-Down Pathway (Kuramoto) ---
        self.enc = nn.Linear(self.latent_dim, self.in_dim)
        
        # Enhanced Projection Head (as per your snippet)
        self.f_proj = nn.Sequential(
            nn.Linear(self.in_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )

        self.kuramoto = GraphVectorKuramoto(
            N=self.in_dim, D=self.osc_dim, K=args.k, dt=args.dt, device=device
        )

        # --- Interaction & Bottom-Up Pathway ---
        self.mask_proj = nn.Linear(self.in_dim, self.in_dim)
        
        self.core = RegionAlignedSNN(
            T=T, 
            num_regions=self.in_dim, 
            input_feat_dim=self.osc_dim, 
            num_classes=num_classes,
            low_n=args.low_n, 
            high_n=args.high_n, 
            branch=args.branch, 
            device=device
        )
        
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, x, sc):
        B, T, N = x.shape
        x = x.to(self.device)
        sc = sc.to(self.device)

        theta = torch.zeros(B, self.in_dim, self.osc_dim, device=self.device)
        theta_hist = []
        
        feats_list = [] 
        mask_hidden_list = [] 

        # === 1. Dual-Stream Dynamics ===
        for t in range(T):
            x_t = x[:, t, :] 
            z_t = self.f_proj(x_t)
            gamma_t = self.enc(z_t)

            theta = self.kuramoto(theta, gamma_t, A=sc) 
            theta_hist.append(theta)

            # Phase Delay & Gating
            idx = max(0, t - self.phase_delay_steps)
            theta_mean = theta_hist[idx].mean(dim=-1) 
            mask_116 = 0.5 * (1.0 + torch.sin(theta_mean)) 

            phase_feat = torch.sin(theta) 
            phase_feat_gated = phase_feat * mask_116.unsqueeze(-1) 
            feats_list.append(phase_feat_gated.unsqueeze(1))

            mask_hidden = torch.sigmoid(self.mask_proj(mask_116))
            mask_hidden_list.append(mask_hidden)

        # === 2. SNN Processing ===
        core_input = torch.cat(feats_list, dim=1)            
        all_hidden_masks = torch.stack(mask_hidden_list, dim=1) 

        core_out, spikes = self.core(core_input, gating_signals=all_hidden_masks)
        
        # === 3. Temporal Pooling (For Subject Classification) ===
        # Pooling: [Batch, Classes, Time] -> [Batch, Classes]
        logits_pooled = torch.mean(core_out, dim=2) 
        
        return self.logsoftmax(logits_pooled), spikes