import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.optim.lr_scheduler import StepLR
from SNN.Hyperparameters import args
from models.s2net import S2Net
from utils.data_loader import BoldSequenceDatasetOneLabelSC, BoldSequenceDatasetOneLabelSC_re
from utils.metrics import compute_metrics 
from utils.lib import set_seed, dump_json

def make_folds(n_samples, n_folds=10, shuffle=True, seed=1111):
    """
    Generates indices for K-Fold Cross-Validation.
    """
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    fold_indices = []
    for k in range(n_folds):
        val_idx = folds[k]
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
        fold_indices.append((train_idx, val_idx))
    return fold_indices

def run_one_fold(fold_id, train_idx, val_idx, combined_dataset, args, T, N, num_classes, fold_save_dir):
    print(f"\n========== Fold {fold_id + 1} Training ==========")
    
    # 1. Prepare DataLoaders
    train_subset = Subset(combined_dataset, train_idx)
    val_subset   = Subset(combined_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=2)
    val_loader   = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2)

    # 2. Initialize S2-Net Model
    # The 'args' object contains hyperparameters for Top-Down and Bottom-Up pathways
    model = S2Net(
        T=T, 
        num_regions=N, 
        num_classes=num_classes, 
        args=args, 
        device=args.device
    ).to(args.device)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.75)

    # 3. Track Best Metrics
    best_metrics = {
        "epoch": 0,
        "val_acc": 0.0,
        "val_prec": 0.0,
        "val_rec": 0.0,
        "val_f1": 0.0,
        "val_auc": 0.0
    }
    
    ckpt_path = os.path.join(fold_save_dir, f'model_fold{fold_id+1}.pt')

    # 4. Training Loop
    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        
        # Ensure SNN masks are reset if required by the implementation
        if hasattr(model.bottom_up, 'rnn_layer'):
             model.bottom_up.rnn_layer.apply_mask()
        
        train_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels, fc in train_loader:
            inputs = inputs.to(args.device).float()
            fc = fc.to(args.device).float() # Structural Connectivity (SC) matrix
            labels = labels.to(args.device).long()
            
            # Align Label dimensions for sequence labeling: [B] -> [B, T]
            if labels.dim() == 1: 
                labels = labels.unsqueeze(1).repeat(1, T)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs, fc) 
            
            # Flatten [B, C, T] -> [B*T, C] for NLLLoss calculation
            B_, C_, T_ = outputs.shape
            outputs_flat = outputs.permute(0, 2, 1).reshape(B_ * T_, C_)
            labels_flat = labels.reshape(B_ * T_)
            
            loss = criterion(outputs_flat, labels_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Re-apply mask after weight update for sparse connectivity maintenance
            if hasattr(model.bottom_up, 'rnn_layer'):
                 model.bottom_up.rnn_layer.apply_mask()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.numel()
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / max(1, total)
        
        # --- Validation Phase ---
        model.eval()
        all_y_flat, all_probs_flat = [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels, fc in val_loader:
                inputs = inputs.to(args.device).float()
                fc = fc.to(args.device).float()
                labels = labels.to(args.device).long()
                if labels.dim() == 1: labels = labels.unsqueeze(1).repeat(1, T)
                
                outputs, spikes = model(inputs, fc)
                
                B_, C_, T_ = outputs.shape
                loss = criterion(outputs.permute(0, 2, 1).reshape(B_ * T_, C_), labels.reshape(B_ * T_))
                val_loss += loss.item()
                
                # Collect predictions for sklearn metrics
                all_y_flat.append(labels.cpu().numpy().flatten()) 
                all_probs_flat.append(outputs.exp().permute(0, 2, 1).reshape(-1, C_).cpu().numpy())

        # Compute Metrics
        y_true = np.concatenate(all_y_flat)
        prob_mat = np.concatenate(all_probs_flat)
        
        acc, prec, rec, f1, auc = compute_metrics(y_true, prob_mat)
        avg_val_loss = val_loss / max(1, len(val_loader))
        
        print(f'Fold {fold_id+1} | Ep {epoch+1} | Loss: {avg_val_loss:.4f} | TrAcc: {train_acc:.2f}% | ValAcc: {acc*100:.2f}% | F1: {f1:.3f}')
        
        scheduler.step()

        # Save Best Model based on Accuracy
        if acc > best_metrics["val_acc"]:
            best_metrics = {
                "epoch": epoch, "val_acc": acc, "val_prec": prec,
                "val_rec": rec, "val_f1": f1, "val_auc": auc
            }
            torch.save(model.state_dict(), ckpt_path)

    return best_metrics

def main():
    set_seed(args.seed)
    print(f'Using device: {args.device}')
    
    # 1. Directory Setup
    exp_name = f'S2Net_hidden{args.hidden}_K{args.k}_lr{args.lr}'
    exp_dir = os.path.join(args.output_dir, exp_name)
    record_dir = os.path.join(exp_dir, 'records')
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    
    os.makedirs(record_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 2. Load Datasets
    print("Loading Datasets...")
    dataset1 = BoldSequenceDatasetOneLabelSC(
        data_dir=args.data_root, 
        label_csv=args.label_csv_rl,
        sc_root_dir=args.sc_root, 
        fallback_sc_path=args.fallback_sc,
        required_T=405, 
        cache_sc=True,
    )
    dataset2 = BoldSequenceDatasetOneLabelSC_re(
        data_dir=args.data_root, 
        label_csv=args.label_csv_lr,
        sc_root_dir=args.sc_root, 
        fallback_sc_path=args.fallback_sc,
        required_T=405, 
        cache_sc=True,
    )

    T, N = dataset1.T, dataset1.N
    
    # 3. Process Label Mapping
    labels1, labels2 = dataset1.labels.numpy(), dataset2.labels.numpy()
    all_labels = np.concatenate([labels1, labels2])
    uniq_vals = np.unique(all_labels)
    num_classes = len(uniq_vals)
    print(f"Dataset info: TimeSteps(T)={T}, Regions(N)={N}, Classes={num_classes} ({uniq_vals})")

    mapping = {v: i for i, v in enumerate(sorted(uniq_vals))}
    dataset1.labels = torch.from_numpy(np.vectorize(mapping.get)(labels1).astype(np.int64))
    dataset2.labels = torch.from_numpy(np.vectorize(mapping.get)(labels2).astype(np.int64))

    combined_dataset = ConcatDataset([dataset1, dataset2])

    # 4. K-Fold Cross Validation
    fold_indices = make_folds(len(combined_dataset), n_folds=args.folds, shuffle=True, seed=args.seed)
    all_metrics = []
    
    for fold_id, (tr_idx, va_idx) in enumerate(fold_indices):
        metrics = run_one_fold(
            fold_id, tr_idx, va_idx, 
            combined_dataset, args, T, N, num_classes, 
            ckpt_dir
        )
        all_metrics.append(metrics)
        
        # Save metrics for current fold
        dump_json(metrics, record_dir, f"fold{fold_id+1}_metrics")

    # 5. Final Summary
    accs = np.array([m["val_acc"] for m in all_metrics])
    f1s = np.array([m["val_f1"] for m in all_metrics])
    
    print(f"\n===== Final {args.folds}-Fold Summary =====")
    print(f"Mean Acc:  {accs.mean()*100:.2f}% +/- {accs.std()*100:.2f}")
    print(f"Mean F1:   {f1s.mean():.4f}")

    overall_results = {
        "mean_acc": float(accs.mean()), "std_acc": float(accs.std()),
        "mean_f1": float(f1s.mean()),   "std_f1": float(f1s.std()),
        "all_folds": all_metrics
    }
    dump_json(overall_results, record_dir, "summary")
    print(f"Results saved to {record_dir}")

if __name__ == "__main__":
    main()