# S²-Net: Synchronized Spiking Neural Network (Anonymous)

**[Supplementary Code for ICML 2026 Submission]**

This repository contains the official PyTorch implementation of **S²-Net**, a brain-inspired framework that integrates **Top-Down Neural Oscillatory Synchronization** with **Bottom-Up Rhythm-Modulated Spiking Neural Networks**.


## 🧠 Model Architecture & Methodology

S²-Net models cognitive dynamics from fMRI BOLD signals by simulating the interaction between macroscopic oscillations and microscopic neuronal firing.

**Core Components:**

1.  **Top-Down Pathway (Vector Kuramoto):**
    * Models macroscopic brain dynamics using Vector-valued Sakaguchi-Kuramoto dynamics.
    * **Dynamics:** The phase evolution follows the topologically constrained system:
        $$\dot{\theta}_i^d(t) = \omega_i^d + \kappa_i^d \sin(\gamma_i(t) - \theta_i^d(t)) + \frac{K}{N} \sum_{j} A_{ij} \sin(\theta_j^d(t) - \theta_i^d(t) - \alpha_{ij}(t))$$

2.  **Interaction Mechanism (Phase Gating):**
    * Introduces a learnable time delay $\tau$ to generate a gating signal $g_i(t)$ from the oscillatory phase:
        $$g_i(t) = \sigma_{\text{gate}}(\sin(\theta_i(t - \tau)))$$
    * This mechanism selectively activates specific phase patterns, dynamically grouping spiking neurons via traveling waves.

3.  **Bottom-Up Pathway (Modulated SNN):**
    * Incorporates dendritic integration where the membrane potential $U_i(t)$ is modulated by the rhythmic gate $g_i(t)$.
    * **Update Rule:**
        $$U_i(t) = (1 - g_i(t)) \cdot U_i(t-1) + g_i(t) \cdot V_i(t)$$

4.  **Phase Lag Inference (Optimal Transport):**
    * Estimates phase lags $\alpha_{ij}(t)$ using entropically regularized Wasserstein Optimal Transport on the graph structure $A$, minimizing the transport cost $C_{ij} \propto 1/(A_{ij} + \epsilon)$.



## 📂 Repository Structure

The codebase is organized to separate core dynamical models from task-specific implementations:

```text
S2Net/
├── models/                  # [S²-Net Architectures] Task-specific model variants
│   ├── s2net.py             # Standard S²-Net for Sequence Labeling (Eq. 2-6)
│   ├── s2net_cls.py         # S²-Net Classifier with Temporal Pooling (Subject Classification)
│   └── s2net_recon.py       # S²-Net Autoencoder (Self-Supervised Reconstruction)
│
├── SNN/                     # [Core Dynamics] Neural dynamics & Kuramoto layers
│   ├── SNN_layers/
│   │   └── vector_kuramoto.p  # Graph-aware Vector Kuramoto with OT phase lag (Eq. 2)
│   ├── spiking_model.py          # Rhythm-modulated Spiking Neural Network (Eq. 6)
│   └── Hyperparameters.py        # Centralized configuration management
│
├── utils/                   # [Utilities] Data handling & Evaluation
│   ├── data_loader.py       # Data loaders for HCP, ADNI, UKB, NIFD datasets
│   ├── metrics.py           # Evaluation metrics (Accuracy, F1, MSE, Intra/Inter Sync)
│   └── lib.py               # Helper functions (Seed, Logging)
│
├── train_s2net_sim.py       # [Experiment 1] Proof-of-Concept Simulation (Synthetic Data)
├── train_s2net_recon.py     # [Experiment 2] Self-Supervised BOLD Reconstruction
├── train_s2net_cls.py       # [Experiment 3] Subject-Level Classification (HCPA/HCPYA/UKB/ADNI/PPMI/NIFD)
└── train_s2net_dyn.py       # [Experiment 4] Dynamic State Recognition (HCP-WM)
```

## 🚀 Usage Instructions

## 1. Proof-of-Concept Simulation
Reproduce the "Binding-by-Synchrony" and "Segregation-by-Desynchrony" experiments on synthetic data (Disjoint Squares). This script validates the core physics without external data.

```bash
python train_s2net_sim.py --output_dir ./exp/simulation
```

## 2. Self-Supervised Reconstruction
Train S²-Net to reconstruct BOLD signals (Autoencoder mode), demonstrating its capability to capture intrinsic brain dynamics.

```bash
python train_s2net_recon.py \
  --dataset HCPYA \
  --data_root /path/to/HCP/BOLD/ \
  --sc_root /path/to/HCP/SC/ \
  --fallback_sc /path/to/fallback.mat
```

## 3. Subject Classification
Apply S²-Net for disease diagnosis or subject classification on datasets like ADNI or PPMI.
```bash
python train_s2net_cls.py \
  --dataset ADNI \
  --data_root /path/to/ADNI/BOLD/ \
  --sc_root /path/to/ADNI/SC/ \
  --label_file /path/to/ADNI/labels.csv
```

## 4. Dynamic State Recognition (Sequence Labeling)
Perform frame-wise cognitive state decoding (e.g., working memory tasks from HCP-WM) on task-fMRI data.

```bash
python train_s2net_dyn.py \
  --dataset HCPYA \
  --data_root /path/to/HCP/BOLD/ \
  --sc_root /path/to/HCP/SC/ \
  --label_csv_rl /path/to/labels_RL.csv \
  --label_csv_lr /path/to/labels_LR.csv
```


## ⚙️ Configuration
Key hyperparameters are managed in SNN/Hyperparameters.py. 
You can override them via command line arguments:

- --k: Global coupling strength $K$ (Eq. 2).
- --dt: Integration time step.
- --hidden: Latent dimension size.
- --low_n / --high_n: Spiking neuron adaptation parameters.
- --ablation: Choose ablation modes (no_coupling, no_mask, raw_feature, full).