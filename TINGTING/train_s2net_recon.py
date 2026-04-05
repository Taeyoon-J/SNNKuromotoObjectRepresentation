import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold

# Local Imports
from models.s2net_recon import S2NetReconstruction
from utils.data_loader import HCPYA, HCPA, UKB, ADNI, PPMI, NIFD
from utils.lib import set_seed, dump_json, count_parameters

# Import shared args
try:
    from SNN.Hyperparameters import args
except ImportError:
    # Fallback if running directly without proper path setup
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from SNN.Hyperparameters import args

def load_dataset_recon(args):
    """Factory function for loading datasets (Recon usually ignores labels)."""
    print(f"Loading Dataset: {args.dataset}...")
    
    if args.dataset == "HCPYA":
        return HCPYA(data_dir=args.data_root, sc_root_dir=args.sc_root, fallback_sc_path=args.fallback_sc, T_fix=220, cache_sc=True)
    elif args.dataset == "HCPA":
        return HCPA(data_dir=args.data_root, sc_root_dir=args.sc_root, fallback_sc_path=args.fallback_sc, T_fix=220, cache_sc=True)
    elif args.dataset == "UKB":
        return UKB(data_dir=args.data_root, sc_root_dir=args.sc_root, fallback_sc_path=args.fallback_sc, T_fix=100, cache_sc=True)
    elif args.dataset == "ADNI":
        # Label file is technically not needed for recon, but dataset class might require it
        return ADNI(data_dir=args.data_root, label_csv=args.label_file, sc_root_dir=args.sc_root, fallback_sc_path=args.fallback_sc, T_fix=200, cache_sc=True)
    elif args.dataset == "PPMI":
        return PPMI(data_dir=args.data_root, label_csv=args.label_file, sc_root_dir=args.sc_root, fallback_sc_path=args.fallback_sc, T_fix=200, cache_sc=True)
    elif args.dataset == "NIFD":
        return NIFD(data_dir=args.data_root, label_xlsx=args.label_file, sc_root_dir=args.sc_root, fallback_sc_path=args.fallback_sc, T_fix=200, cache_sc=True)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

def run_one_fold(fold_id, train_idx, val_idx, dataset, args, dirs):
    print(f"\n========== Fold {fold_id + 1} ==========")
    device = args.device
    
    train_subset = Subset(dataset, train_idx)
    val_subset   = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=2)
    val_loader   = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2)

    # 1. Model Setup
    model = S2NetReconstruction(
        T=dataset.T, num_regions=dataset.N, args=args, device=device
    ).to(device)
    
    # print(f"Params: {count_parameters(model)}")

    criterion = nn.MSELoss()
    # Reconstruction often benefits from higher initial LR
    recon_lr = 0.005 if args.lr == 1e-3 else args.lr 
    optimizer = torch.optim.Adam(model.parameters(), lr=recon_lr)
    
    # Scheduler: Reduce LR when recon loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 2. Tracking
    best_metrics = {"val_mse": float('inf'), "val_mae": float('inf'), "epoch": 0}
    
    fold_base = f"{args.dataset}_fold{fold_id+1}"
    ckpt_path = os.path.join(dirs['ckpt'], f"{fold_base}.pt")
    spike_path = os.path.join(dirs['spikes'], f"spikes_{fold_base}_best.npz")

    # 3. Loop
    for epoch in range(args.epochs):
        # Train
        model.train()
        if hasattr(model.core.rnn_layer, 'apply_mask'): model.core.rnn_layer.apply_mask()
        
        train_loss = 0.0
        for inputs, _, sc in train_loader:
            inputs = inputs.to(device).float()
            sc = sc.to(device).float()
            
            optimizer.zero_grad()
            recon_outputs, _ = model(inputs, sc) # recon_outputs: [B, T, N]
            
            # MSE Loss against input
            loss = criterion(recon_outputs, inputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if hasattr(model.core.rnn_layer, 'apply_mask'): model.core.rnn_layer.apply_mask()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_mse_sum, val_mae_sum = 0.0, 0.0
        n_batches = 0
        spike_buffer = []
        
        with torch.no_grad():
            for inputs, _, sc in val_loader:
                inputs, sc = inputs.to(device).float(), sc.to(device).float()
                
                recon_outputs, spikes = model(inputs, sc)
                
                mse = F.mse_loss(recon_outputs, inputs).item()
                mae = F.l1_loss(recon_outputs, inputs).item()
                
                val_mse_sum += mse
                val_mae_sum += mae
                n_batches += 1
                spike_buffer.append(spikes.cpu().numpy().astype(bool))

        avg_val_mse = val_mse_sum / max(1, n_batches)
        avg_val_mae = val_mae_sum / max(1, n_batches)
        
        # Log
        if (epoch+1) % 10 == 0 or epoch==0:
            print(f"Fold {fold_id+1} | Ep {epoch+1} | TrLoss: {train_loss/len(train_loader):.4f} | Val MSE: {avg_val_mse:.4f}")
        
        # Step Scheduler
        scheduler.step(avg_val_mse)

        # Save Best
        if avg_val_mse < best_metrics["val_mse"]:
            best_metrics = {"epoch": epoch, "val_mse": avg_val_mse, "val_mae": avg_val_mae}
            torch.save(model.state_dict(), ckpt_path)
            
            full_spikes = np.concatenate(spike_buffer, axis=0)
            np.savez_compressed(spike_path, spikes=full_spikes)
            print(f"  --> New Best! MSE: {avg_val_mse:.4f}")

    return best_metrics

def main():
    set_seed(args.seed)
    print(f"Device: {args.device} | Dataset: {args.dataset}")
    
    # 1. Directories
    exp_name = f"{args.dataset}_Recon_hidden{args.hidden}"
    exp_dir = os.path.join(args.output_dir, exp_name)
    dirs = {
        'ckpt': os.path.join(exp_dir, "checkpoint"),
        'rec': os.path.join(exp_dir, "record"),
        'spikes': os.path.join(exp_dir, "spikes")
    }
    for d in dirs.values(): os.makedirs(d, exist_ok=True)

    # 2. Data
    dataset = load_dataset_recon(args)
    print(f"Data Loaded: N={len(dataset)}, Regions={dataset.N}, Time={dataset.T}")

    # 3. K-Fold (Using Dummy labels for random stratified split if needed, or just random)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    y_dummy = np.zeros(len(dataset)) 
    fold_indices = list(skf.split(y_dummy, y_dummy))
    
    all_metrics = []
    
    for fold_id, (tr_idx, va_idx) in enumerate(fold_indices):
        m = run_one_fold(fold_id, tr_idx, va_idx, dataset, args, dirs)
        all_metrics.append(m)
        dump_json(m, dirs['rec'], f"{args.dataset}_fold{fold_id+1}_metrics")

    # 4. Summary
    mses = np.array([m["val_mse"] for m in all_metrics])
    maes = np.array([m["val_mae"] for m in all_metrics])
    
    print(f"\n===== Final {args.folds}-Fold Reconstruction Summary =====")
    print(f"Mean MSE: {mses.mean():.5f} +/- {mses.std():.5f}")
    
    overall = {
        "mean_mse": float(mses.mean()), "std_mse": float(mses.std()),
        "mean_mae": float(maes.mean()), "all_folds": all_metrics
    }
    dump_json(overall, dirs['rec'], f"{args.dataset}_summary")
    print(f"Results saved to {dirs['rec']}")

if __name__ == "__main__":
    main()