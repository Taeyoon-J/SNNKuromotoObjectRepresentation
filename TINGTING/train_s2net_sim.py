import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if "--data_root" not in sys.argv:
    sys.argv.extend([
        "--data_root", "./dummy_data_root",
        "--sc_root", "./dummy_sc_root",
        "--label_file", "./dummy_label.csv"
    ])

# ================= 1. CONFIG =================
CONFIG = {
    "batch_size": 16,
    "epochs": 100,
    "lr": 0.001,
    "device": "cuda",
    "save_dir": "./toy_simulation_logs",
    "img_size": 28,
    "kuramoto_steps": 60,
    "dt": 0.1,
    "coupling_K": 18.0,
    "drive_kappa": 2.5,
    "lag_scale": 1.5,
    "gating_delay": 12,
    "neg_weight": 2.0,
    "seed": 123,
    "dataset_size": 200,
    "noise_std": 0.05,
    "num_workers": 0
}
#We use grid search to find the optimal parameters#We use grid search to find the optimal parameters. We will release the best parameters after acceptance.
def resolve_device(dev_str: str):
    if dev_str == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev_str)

CONFIG["device"] = resolve_device(CONFIG["device"])
os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ================= 2. Toy Dataset =================
class ToyDataset(Dataset):
    def __init__(self, size=200, img_size=28, noise=0.05):
        self.size = size
        self.img_size = img_size
        self.noise = noise

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = torch.zeros(1, self.img_size, self.img_size)
        mask = torch.zeros(self.img_size, self.img_size, dtype=torch.long)
        # Object 1
        img[:, 5:15, 5:15] = 1.0
        mask[5:15, 5:15] = 1
        # Object 2
        img[:, 15:25, 15:25] = 0.5
        mask[15:25, 15:25] = 2
        img += torch.randn_like(img) * self.noise
        return img, mask
class NestedToyDataset(Dataset):
    def __init__(self, size=200, img_size=28, noise=0.2):
        self.size = size
        self.img_size = img_size
        self.noise = noise

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = torch.zeros(1, self.img_size, self.img_size)
        mask = torch.zeros(self.img_size, self.img_size, dtype=torch.long)
        center = self.img_size // 2

        # Object 1
        img[:, center-3:center+3, center-3:center+3] = 1.0
        mask[center-3:center+3, center-3:center+3] = 1

        # Object 2
        y, x = torch.meshgrid(torch.arange(self.img_size), torch.arange(self.img_size), indexing='ij')
        dist_sq = (x - center)**2 + (y - center)**2
        ring_mask = (dist_sq > 6**2) & (dist_sq < 11**2)
        img[0, ring_mask] = 0.7
        mask[ring_mask] = 2

        img += torch.randn_like(img) * self.noise
        return img, mask
# ================= 3. GraphVectorKuramoto =================
class graphVectorKuramoto(nn.Module):
    """
    Graph-Aware Vector Kuramoto with OT-derived Phase Lags.
    Strictly aligns with Eq. (5) and OT Surrogate mechanics.
    """
    def __init__(self, N, D=2, K=1.0, dt=1.0, alpha_scale=1.0, device="cuda"):
        super().__init__()
        self.N = N
        self.D = D
        self.K = K
        self.dt = dt
        self.alpha_scale = alpha_scale # alpha_0 in paper
        self.device = device

        # Natural frequency ω_i (aligns with revised Eq. 5)
        self.omega = nn.Parameter(torch.randn(N, D) * 0.1)

        # Control stiffness κ_i
        self.kappa = nn.Parameter(torch.ones(N, D))
        self.direction_learner = nn.Parameter(torch.randn(N, N) * 0.01)
        # REMOVED: self.alpha = nn.Parameter(...) 
        # Reason: alpha must be derived from A, not learned freely.

    def forward(self, theta_prev, gamma, A=None):
        """
        A : [B, H, H] Connectivity matrix (Structural priors)
        """
        B, H, D = theta_prev.shape
        device = theta_prev.device

        # 1. Handle Graph Structure & OT Surrogate
        if A is None:
            # Fallback if no graph provided
            A_lat = torch.ones(B, H, H, device=device)
            alpha = torch.zeros(B, H, H, 1, device=device)
        else:
            # Symmetrize connectivity
            A_lat = 0.5 * (A + A.transpose(1, 2))
            A_lat = torch.relu(A_lat) + 1e-6 # Avoid div by zero

            # --- OT-Derived Phase Lag (Section 3.3 in Paper) ---
            # Cost C_ij = 1 / A_ij
            cost_matrix = 1.0 / A_lat 
            
            # Normalize cost to [0, 1] per batch to stabilize
            c_min = cost_matrix.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
            c_max = cost_matrix.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
            norm_cost = (cost_matrix - c_min) / (c_max - c_min + 1e-6)
            direction = self.direction_learner - self.direction_learner.transpose(0, 1) # [N, N]
            direction_mask = torch.tanh(direction)
            alpha_matrix = direction_mask.unsqueeze(0) * norm_cost # [B, N, N]
            # alpha_ij = alpha_0 * norm(C_ij)
            # Expand to [B, H, H, 1] to broadcast over D dim
            # alpha = (self.alpha_scale * norm_cost).unsqueeze(-1)
            alpha = (self.alpha_scale * alpha_matrix).unsqueeze(-1)

        # 2. Kuramoto Dynamics
        theta_i = theta_prev.unsqueeze(2) # [B, H, 1, D]
        theta_j = theta_prev.unsqueeze(1) # [B, 1, H, D]

        # Interaction term: sin(theta_j - theta_i - alpha_ij)
        phase_diff = theta_j - theta_i - alpha
        
        # Weighted sum by adjacency A_ij
        interaction = torch.sum(A_lat.unsqueeze(-1) * torch.sin(phase_diff), dim=2)
        coupling = (self.K / float(H)) * interaction

        # 3. Sensory Drive (Corrected to Sinusoidal)
        # kappa * sin(gamma - theta)
        # gamma is [B, N, D] or [B, N, 1] -> unsqueeze if needed
        if gamma.dim() == 2:
            gamma_exp = gamma.unsqueeze(-1)
        else:
            gamma_exp = gamma
            
        drive_term = self.kappa * torch.sin(gamma_exp - theta_prev)

        # 4. Euler Integration
        theta_dot = self.omega + coupling + drive_term
        theta_new = theta_prev + self.dt * theta_dot
        
        return theta_new

# ================= 4. Wrapper Model =================
class GraphOTKuramotoWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.H = cfg['img_size']
        self.W = cfg['img_size']
        self.N = self.H * self.W
        
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16), nn.Tanh(),
            nn.Conv2d(16, 16, 3, padding=1), nn.Tanh()
        )
        self.feature_head = nn.Conv2d(16, 8, 1)
        self.drive_head = nn.Conv2d(16, 1, 1)

        # graphVectorKuramoto
        self.kuramoto_layer = graphVectorKuramoto(
            N=self.N,
            D=1,  
            K=float(cfg['coupling_K']),
            dt=float(cfg['dt']),
            alpha_scale=float(cfg['lag_scale']),
            device=cfg['device']
        )

    def forward(self, x):
        cfg = self.cfg
        B = x.shape[0]
        shared_feat = self.backbone(x)

        #  Gamma (Target)
        drive_out = self.drive_head(shared_feat) 
        gamma = drive_out.view(B, self.N, 1)    

        # Connectivity A
        feat = self.feature_head(shared_feat).view(B, 8, self.N).permute(0, 2, 1)
        feat_norm = F.normalize(feat, p=2, dim=2)
        A = torch.bmm(feat_norm, feat_norm.transpose(1, 2))
        eye = torch.eye(self.N, device=x.device).unsqueeze(0)
        A = A * (1 - eye)

        # evolution
        theta = torch.rand(B, self.N, 1, device=x.device) * 2 * np.pi

        for t in range(cfg['kuramoto_steps']):
            theta = self.kuramoto_layer(theta, gamma, A=A)

        return theta.squeeze(-1)

# ================= 5. Loss & Metrics =================
def phase_contrastive_loss(theta, mask, neg_weight=1.0):
    B, N = theta.shape
    mask_flat = mask.view(B, N)
    loss = 0.0
    valid_batch_count = 0
    for i in range(B):
        valid = mask_flat[i] > 0
        if valid.sum() < 2: continue
        th = theta[i, valid]
        lbl = mask_flat[i, valid]
        cos_sim = torch.cos(th.unsqueeze(0) - th.unsqueeze(1))
        label_mat = (lbl.unsqueeze(0) == lbl.unsqueeze(1)).float()
        pos = ((1 - cos_sim) * label_mat).sum() / (label_mat.sum() + 1e-8)
        neg_mask = 1 - label_mat
        neg = ((1 + cos_sim) * neg_mask).sum() / (neg_mask.sum() + 1e-8)
        loss = loss + pos + neg_weight * neg
        valid_batch_count += 1
    if valid_batch_count == 0: return torch.tensor(0.0, device=theta.device, requires_grad=True)
    return loss / valid_batch_count

def calc_metrics(theta, mask):
    with torch.no_grad():
        th = theta[0].detach().cpu().numpy()
        m = mask[0].view(-1).cpu().numpy()
        th1 = th[m == 1]
        th2 = th[m == 2]
        if len(th1) > 0: z1 = np.mean(np.exp(1j * th1))
        else: z1 = 0
        if len(th2) > 0: z2 = np.mean(np.exp(1j * th2))
        else: z2 = 0
        R1 = np.abs(z1)
        R2 = np.abs(z2)
        intra = (R1 + R2) / 2.0
        phi1 = np.angle(z1)
        phi2 = np.angle(z2)
        dphi = phi1 - phi2
        inter = 0.5 * (1.0 + np.cos(dphi))
        return intra, inter

# ================= 6. Main Loop =================
def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_simulation():
    cfg = CONFIG
    seed_all(cfg["seed"])
    print(f"Device: {cfg['device']}")
   
    dataset = ToyDataset(size=cfg["dataset_size"], img_size=cfg["img_size"], noise=cfg["noise_std"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], drop_last=True)

    model = GraphOTKuramotoWrapper(cfg).to(cfg["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    print(f"Start simulation using explicit 'graphVectorKuramoto' class...")
    print(f"Epochs: {cfg['epochs']}, Steps: {cfg['kuramoto_steps']}")
    start_time = time.time()

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = []
        last_theta = None
        last_mask = None

        for bx, by in loader:
            bx = bx.to(cfg["device"])
            by = by.to(cfg["device"])

            theta_final = model(bx)
            loss = phase_contrastive_loss(theta_final, by, neg_weight=cfg["neg_weight"])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss.append(loss.item())
            last_theta = theta_final
            last_mask = by

        avg_loss = np.mean(epoch_loss)
        intra, inter = calc_metrics(last_theta, last_mask)
        
        print(f"Epoch [{epoch+1}/{cfg['epochs']}], loss={avg_loss:.4f}, intra={intra:.3f}, inter={inter:.3f}")

    total_time = time.time() - start_time
    print(f"Simulation finished in {total_time:.2f} seconds.")

if __name__ == "__main__":
    run_simulation()