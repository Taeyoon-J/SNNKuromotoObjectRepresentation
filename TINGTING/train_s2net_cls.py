import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import StratifiedKFold

# Local Imports
from models.s2net_cls import S2NetClassifier
from utils.metrics import compute_metrics
from utils.data_loader import HCPYA, HCPA, UKB, ADNI, PPMI, NIFD
from utils.lib import set_seed, dump_json
from SNN.Hyperparameters import args
def load_dataset(args):
    """Factory function to load specific datasets based on arguments."""
    print(f"Loading Dataset: {args.dataset}...")
    
    if args.dataset == "HCPYA":
        return HCPYA(data_dir=args.data_root, sc_root_dir=args.sc_root, fallback_sc_path=args.fallback_sc, T_fix=220, cache_sc=True)
    elif args.dataset == "HCPA":
        return HCPA(data_dir=args.data_root, sc_root_dir=args.sc_root, fallback_sc_path=args.fallback_sc, T_fix=220, cache_sc=True)
    elif args.dataset == "UKB":
        return UKB(data_dir=args.data_root, sc_root_dir=args.sc_root, fallback_sc_path=args.fallback_sc, T_fix=100, cache_sc=True)
    elif args.dataset == "ADNI":
        if not args.label_file: raise ValueError("ADNI requires --label_file")
        return ADNI(data_dir=args.data_root, label_csv=args.label_file, sc_root_dir=args.sc_root, fallback_sc_path=args.fallback_sc, T_fix=200, cache_sc=True)
    elif args.dataset == "PPMI":
        if not args.label_file: raise ValueError("PPMI requires --label_file")
        return PPMI(data_dir=args.data_root, label_csv=args.label_file, sc_root_dir=args.sc_root, fallback_sc_path=args.fallback_sc, T_fix=200, cache_sc=True)
    elif args.dataset == "NIFD":
        if not args.label_file: raise ValueError("NIFD requires --label_file (xlsx)")
        return NIFD(data_dir=args.data_root, label_xlsx=args.label_file, sc_root_dir=args.sc_root, fallback_sc_path=args.fallback_sc, T_fix=200, cache_sc=True)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

def get_all_labels(dataset):
    """Helper to extract all labels for StratifiedKFold."""
    ys = []
    for i in range(len(dataset)):
        _, y, _ = dataset[i]
        # Handle tensor or scalar label
        if torch.is_tensor(y):
            y = y.detach().cpu()
            y = int(y.item()) if y.numel() == 1 else int(y.view(-1)[0].item())
        else:
            y = int(y)
        ys.append(y)
    return np.asarray(ys, dtype=int)

def run_one_fold(fold_id, train_idx, val_idx, dataset, y_all, args, num_classes, dirs):
    print(f"\n========== Fold {fold_id + 1} ==========")
    device = args.device
    
    # 1. Prepare Weighted Sampler for Class Imbalance
    train_subset = Subset(dataset, train_idx)
    val_subset   = Subset(dataset, val_idx)
    
    y_tr = y_all[train_idx]
    class_counts = np.bincount(y_tr, minlength=int(y_all.max()) + 1)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y_tr]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double), 
        num_samples=len(sample_weights), 
        replacement=True
    )

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    val_loader   = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 2. Model Setup
    model = S2NetClassifier(
        T=dataset.T, num_regions=dataset.N, num_classes=num_classes, args=args, device=device
    ).to(device)
    
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.75)

    # 3. Metrics & Paths
    best_metrics = {k: 0.0 for k in ["val_acc", "val_prec", "val_rec", "val_f1", "val_auc"]}
    best_metrics["epoch"] = 0
    
    fold_base = f"{args.dataset}_fold{fold_id+1}"
    ckpt_path = os.path.join(dirs['ckpt'], f"{fold_base}.pt")
    spike_path = os.path.join(dirs['spikes'], f"spikes_{fold_base}_best.npz")

    # 4. Training Loop
    for epoch in range(args.epochs):
        # Train
        model.train()
        if hasattr(model.core.rnn_layer, 'apply_mask'): model.core.rnn_layer.apply_mask()
        
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels, sc in train_loader:
            inputs = inputs.to(device).float()
            sc = sc.to(device).float()
            labels = labels.to(device).long().view(-1)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs, sc)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if hasattr(model.core.rnn_layer, 'apply_mask'): model.core.rnn_layer.apply_mask()

            train_loss += loss.item()
            pred = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            
        train_acc = 100.0 * correct / max(1, total)

        # Validation
        model.eval()
        val_loss = 0.0
        all_lbls, all_probs, spike_buffer = [], [], []
        
        with torch.no_grad():
            for inputs, labels, sc in val_loader:
                inputs, sc = inputs.to(device).float(), sc.to(device).float()
                labels = labels.to(device).long().view(-1)
                
                outputs, spikes = model(inputs, sc)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                all_lbls.append(labels.cpu().numpy())
                all_probs.append(outputs.exp().cpu().numpy())
                spike_buffer.append(spikes.cpu().numpy().astype(bool))

        y_true = np.concatenate(all_lbls)
        probs = np.concatenate(all_probs)
        acc, prec, rec, f1, auc = compute_metrics(y_true, probs)
        
        avg_val_loss = val_loss / max(1, len(val_loader))
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Fold {fold_id+1} | Ep {epoch+1} | Loss: {avg_val_loss:.4f} | Val Acc: {acc*100:.2f}% | F1: {f1:.3f}")

        scheduler.step()

        # Save Best
        if acc > best_metrics["val_acc"]:
            best_metrics = {
                "epoch": epoch, "val_acc": acc, "val_prec": prec, 
                "val_rec": rec, "val_f1": f1, "val_auc": auc
            }
            torch.save(model.state_dict(), ckpt_path)
            
            # Save Spikes (Compressed)
            full_spikes = np.concatenate(spike_buffer, axis=0)
            np.savez_compressed(spike_path, spikes=full_spikes, labels=y_true, probs=probs)
            print(f"  --> New Best! Acc: {acc*100:.2f}% | Spikes saved.")

    return best_metrics

def main():
    set_seed(args.seed)
    print(f"Device: {args.device} | Dataset: {args.dataset}")
    
    # 1. Setup Directories
    exp_name = f"{args.dataset}_SubjCLS_hidden{args.hidden}_lr{args.lr}"
    exp_dir = os.path.join(args.output_dir, exp_name)
    dirs = {
        'ckpt': os.path.join(exp_dir, "checkpoint"),
        'rec': os.path.join(exp_dir, "record"),
        'spikes': os.path.join(exp_dir, "spikes")
    }
    for d in dirs.values(): os.makedirs(d, exist_ok=True)

    # 2. Load Dataset
    dataset = load_dataset(args)
    y_all = get_all_labels(dataset)
    num_classes = getattr(dataset, "num_classes", len(np.unique(y_all)))
    print(f"Data Loaded: N={len(dataset)}, Regions={dataset.N}, Time={dataset.T}, Classes={num_classes}")

    # 3. Stratified K-Fold
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_indices = list(skf.split(np.zeros(len(y_all)), y_all))
    
    all_metrics = []
    
    for fold_id, (tr_idx, va_idx) in enumerate(fold_indices):
        m = run_one_fold(fold_id, tr_idx, va_idx, dataset, y_all, args, num_classes, dirs)
        all_metrics.append(m)
        dump_json(m, dirs['rec'], f"{args.dataset}_fold{fold_id+1}_metrics")

    # 4. Summary
    accs = np.array([m["val_acc"] for m in all_metrics])
    f1s = np.array([m["val_f1"] for m in all_metrics])
    
    print(f"\n===== Final {args.folds}-Fold Summary ({args.dataset}) =====")
    print(f"Mean Acc:  {accs.mean()*100:.2f}% +/- {accs.std()*100:.2f}")
    
    overall = {
        "mean_acc": float(accs.mean()), "std_acc": float(accs.std()),
        "mean_f1": float(f1s.mean()), "all_folds": all_metrics
    }
    dump_json(overall, dirs['rec'], f"{args.dataset}_summary")
    print(f"Results saved to {dirs['rec']}")

if __name__ == "__main__":
    main()