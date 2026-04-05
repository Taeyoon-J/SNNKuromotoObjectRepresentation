import torch
import torch.nn as nn
from .layers import GraphVectorKuramoto, RegionAlignedSNN

class S2NetReconstruction(nn.Module):
    """
    S²-Net for Self-Supervised BOLD Reconstruction.
    
    Task: Reconstruct the input BOLD signal from the Spiking Latent State.
    Structure: Encoder -> Kuramoto -> SNN -> Decoder -> BOLD_hat
    """
    def __init__(self, T, num_regions, args, device="cuda"):
        super().__init__()
        self.T = T
        self.in_dim = int(num_regions)
        self.osc_dim = 4
        self.phase_delay_steps = 2
        self.latent_dim = args.hidden # usually 64 for recon

        # --- Top-Down Pathway ---
        self.enc = nn.Linear(self.latent_dim, self.in_dim)
        
        # Encoder (Bottleneck)
        self.f_proj = nn.Sequential(
            nn.Linear(self.in_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1)
        )

        self.kuramoto = GraphVectorKuramoto(
            N=self.in_dim, D=self.osc_dim, K=args.k, dt=args.dt, device=device
        )

        # --- Interaction ---
        self.mask_proj = nn.Linear(self.in_dim, self.in_dim)

        # --- Bottom-Up Pathway (SNN) ---
        # Note: For reconstruction, the SNN output dimension equals the number of regions (N), not classes.
        self.core = RegionAlignedSNN(
            T=T, 
            num_regions=self.in_dim, 
            input_feat_dim=self.osc_dim, 
            num_classes=self.in_dim, # Output dim = Input dim (N)
            low_n=args.low_n, 
            high_n=args.high_n, 
            branch=args.branch, 
            device=device
        )
        
        # --- Decoder (Reconstruction Head) ---
        # Maps SNN hidden states back to BOLD signal space
        self.decoder = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim * 2), 
            nn.LayerNorm(self.in_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(self.in_dim * 2, self.in_dim * 2),
            nn.LayerNorm(self.in_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Linear(self.in_dim * 2, self.in_dim) 
        )

        self.device = device

    def forward(self, x, sc):
        B, T, N = x.shape
        x = x.to(self.device)
        sc = sc.to(self.device)

        theta = torch.zeros(B, self.in_dim, self.osc_dim, device=self.device)
        theta_hist = []
        
        feats_list = [] 
        mask_hidden_list = [] 

        # === 1. Dynamics ===
        for t in range(T):
            x_t = x[:, t, :] 
            z_t = self.f_proj(x_t)
            gamma_t = self.enc(z_t)

            theta = self.kuramoto(theta, gamma_t, A=sc) 
            theta_hist.append(theta)

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

        # core_out: [Batch, N, Time]
        core_out, spikes = self.core(core_input, gating_signals=all_hidden_masks)
        
        # === 3. Decoding ===
        # Permute for Linear layer: [B, N, T] -> [B, T, N]
        snn_state = core_out.permute(0, 2, 1) 
        recon_bold = self.decoder(snn_state)
        
        # Permute back: [B, T, N] -> [B, N, T] if needed, or keep as [B, T, N] to match input x
        # Let's match input shape x: [B, T, N]
        
        return recon_bold, spikes