# Import the necessary modules

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import vit_b_16, convnext_base, resnet50, densenet121, swin_t
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    average_precision_score
)
import os
import re
from collections import defaultdict
import pandas as pd
import logging
import time
from datetime import datetime
import random
from PIL import Image

# ── Logging setup ─────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()          # also prints to console
    ]
)
logger = logging.getLogger(__name__)
# ──────────────────────────────────────────────────────────────────────────────


# ============================================================================
# MODEL LOADERS
# ============================================================================

def load_vit(num_classes, device):
    """Load Vision Transformer (ViT-B/16) - from scratch"""
    model = vit_b_16(pretrained=False, num_classes=num_classes)
    model = model.to(device)
    return model

def load_convnext(num_classes, device):
    """Load ConvNeXt-Base - from scratch"""
    model = convnext_base(pretrained=False, num_classes=num_classes)
    model = model.to(device)
    return model

def load_resnet(num_classes, device):
    """Load ResNet-50 - from scratch"""
    model = resnet50(pretrained=False, num_classes=num_classes)
    model = model.to(device)
    return model

def load_densenet(num_classes, device):
    """Load DenseNet-121 - from scratch"""
    model = densenet121(pretrained=False, num_classes=num_classes)
    model = model.to(device)
    return model

def load_swin(num_classes, device):
    """
    Load Swin architecture
    Training from scratch
    """
    model = swin_t(pretrained=False, num_classes=num_classes)
    model = model.to(device)
    return model

# ============================================================================
# DATASET transform
# ============================================================================

class TransformedDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        img = self.transform(img)
        return img, label

# Common parameters
TARGET_SIZE = 224
RESIZE_SIZE = 256
INTERPOLATION = InterpolationMode.BILINEAR 
num_classes = 2
num_epochs = 5 
N_FOLDS    = 3   # number of cross-validation folds

# Standard ImageNet transform
transform = transforms.Compose([
    transforms.Resize(RESIZE_SIZE, interpolation=INTERPOLATION),
    transforms.CenterCrop(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])


model_configs = [
    ("ViT",         load_vit),
    ("ConvNeXt",    load_convnext),
    ("ResNet50",    load_resnet),
    ("DenseNet121", load_densenet),
    ("Swin",        load_swin),
]


# ── Training ───────────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, device,
                data_label, model_name, fold_idx,
                num_epochs=num_epochs):
    """
    Train linear head only.
    Checkpoint is saved when *validation macro-AUC* improves
    Returns: (best_model, training_elapsed_seconds)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn   = nn.CrossEntropyLoss()
 
    best_val_auc = -1.0         
 
    save_dir = "Best_trained_models"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(
        save_dir, f"best_{model_name}_{data_label}_fold{fold_idx}.pth"
    )
 
    train_start = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
 
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
 
        avg_train_loss = train_loss / len(train_loader)
 
        # Evaluate on validation set
        _, _, val_auc, _, _, _, _, _, _ = evaluate(model, val_loader, device)
 
        epoch_secs = time.time() - epoch_start
        logger.info(
            f"[{model_name}] Fold {fold_idx} | Epoch {epoch+1}/{num_epochs}  "
            f"Train Loss: {avg_train_loss:.4f}  Val Macro-AUC: {val_auc:.4f}  "
            f"Epoch Time: {epoch_secs:.1f}s"
        )
 
        # ── Checkpoint on best validation macro-AUC ──────────────────────
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
 
    train_elapsed = time.time() - train_start
    model.load_state_dict(torch.load(best_model_path))
    return model, train_elapsed
    

# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(model, dataloader, device):
    """
    Returns: acc, f1_macro, auc_roc, f1_micro, f1_weighted, auprc,
             y_true, y_pred, y_scores
    """
    model.eval()
    y_true, y_pred, y_scores = [], [], []
 
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs  = F.softmax(logits, dim=1)
            preds  = torch.argmax(logits, dim=1)
 
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())
 
    y_true   = np.array(y_true)
    y_pred   = np.array(y_pred)
    y_scores = np.array(y_scores)
 
    acc         = accuracy_score(y_true, y_pred)
    f1_macro    = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    f1_micro    = f1_score(y_true, y_pred, average='micro',    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
 
    # AUC-ROC (macro OvR)
    if y_scores.shape[1] == 2:
        auc_roc = roc_auc_score(y_true, y_scores[:, 1])
    else:
        auc_roc = roc_auc_score(y_true, y_scores, multi_class='ovr', average='macro')
 
    # AUPRC (per-class then macro-average)
    auprc_per_class = []
    for c in range(y_scores.shape[1]):
        binary_labels = (y_true == c).astype(int)
        ap = average_precision_score(binary_labels, y_scores[:, c])
        auprc_per_class.append(ap)
    auprc = float(np.mean(auprc_per_class))
 
    return acc, f1_macro, auc_roc, f1_micro, f1_weighted, auprc, y_true, y_pred, y_scores
 
 # ── Per-fold detailed logging ──────────────────────────────────────────────────
def log_model_results(data_label, model_name, fold_idx, class_names,
                      y_true, y_pred, y_scores,
                      acc, f1_macro, auc_roc, f1_micro, f1_weighted, auprc,
                      train_time_secs, inference_time_secs):
    """Write confusion matrix, classification report and predictions to the log."""
    sep = "=" * 70
    logger.info(f"\n{sep}")
    logger.info(f"RESULTS FOR MODEL: {model_name}  |  Fold {fold_idx}")
    logger.info(sep)
 
    # ── Timing ────────────────────────────────────────────────────────────
    logger.info(f"  Training time     : {train_time_secs:.1f}s  "
                f"({train_time_secs/60:.2f} min)")
    logger.info(f"  Inference time    : {inference_time_secs:.3f}s  "
                f"({inference_time_secs*1000/len(y_true):.2f} ms/sample)")
 
    # ── Summary metrics ────────────────────────────────────────────────────
    logger.info(f"  Accuracy          : {acc:.4f}")
    logger.info(f"  F1 (macro)        : {f1_macro:.4f}")
    logger.info(f"  F1 (micro)        : {f1_micro:.4f}")
    logger.info(f"  F1 (weighted)     : {f1_weighted:.4f}")
    logger.info(f"  AUC-ROC           : {auc_roc:.4f}")
    logger.info(f"  AUPRC (macro)     : {auprc:.4f}")
 
    # ── Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    logger.info("\nConfusion Matrix:")
    header = "Pred → " + "  ".join(f"{c:>10}" for c in class_names)
    logger.info(header)
    for i, row in enumerate(cm):
        row_str = f"True {class_names[i]:>5}: " + "  ".join(f"{v:>10}" for v in row)
        logger.info(row_str)
 
    # ── Per-class classification report ───────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    logger.info("\nClassification Report (per class):")
    logger.info(report)
 
    # ── Per-class AUPRC ────────────────────────────────────────────────────
    logger.info("Per-class AUPRC:")
    for c, name in enumerate(class_names):
        binary_labels = (y_true == c).astype(int)
        ap = average_precision_score(binary_labels, y_scores[:, c])
        logger.info(f"  {name}: {ap:.4f}")
 
    # ── Final predictions table (first 50 rows) ────────────────────────────
    logger.info("\nFinal Predictions (sample — first 50):")
    logger.info(f"  {'Index':>6}  {'True':>10}  {'Pred':>10}  " +
                "  ".join(f"P({c})" for c in class_names))
    for i in range(min(50, len(y_true))):
        score_str = "  ".join(f"{y_scores[i, c]:.4f}" for c in range(len(class_names)))
        logger.info(f"  {i:>6}  {class_names[y_true[i]]:>10}  "
                    f"{class_names[y_pred[i]]:>10}  {score_str}")
 
    if len(y_true) > 50:
        logger.info(f"  ... ({len(y_true) - 50} more rows — see CSV for full predictions)")
 
    # ── Save per-fold predictions to CSV ──────────────────────────────────
    pred_dir = "predictions"
    os.makedirs(pred_dir, exist_ok=True)
    pred_df = pd.DataFrame({
        "index":      range(len(y_true)),
        "true_label": [class_names[t] for t in y_true],
        "pred_label": [class_names[p] for p in y_pred],
        **{f"prob_{c}": y_scores[:, i] for i, c in enumerate(class_names)}
    })
    pred_csv = os.path.join(pred_dir, f"{data_label}_{model_name}_fold{fold_idx}_predictions.csv")
    pred_df.to_csv(pred_csv, index=False)
    logger.info(f"\nFull predictions saved to: {pred_csv}")
    logger.info(sep + "\n")
 
 
# ── Aggregate fold results into mean ± SD ─────────────────────────────────────
def aggregate_folds(fold_metrics: list[dict]) -> dict:
    """
    Given a list of per-fold metric dicts, return a dict with
    {metric: (mean, std)} for every numeric key.
    """
    keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float))]
    aggregated = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics])
        aggregated[k] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
    return aggregated
 
def log_mean_sd_summary(model_name: str, aggregated: dict) -> None:
    """Log mean ± SD across folds for a single model."""
    sep = "─" * 70
    logger.info(f"\n{sep}")
    logger.info(f"CROSS-VALIDATION SUMMARY  |  Model: {model_name}  |  {N_FOLDS} folds")
    logger.info(sep)
    for metric, (mean, sd) in aggregated.items():
        logger.info(f"  {metric:<25}: {mean:.4f} ± {sd:.4f}")
    logger.info(sep)

def valid_file(path):
    return not path.endswith(".DS_Store")

def stratified_subset(dataset, fraction=0.1, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Group samples by class
    class_to_samples = {}
    for idx, (path, label) in enumerate(dataset.samples):
        class_to_samples.setdefault(label, []).append(idx)
    
    # Sample fraction from each class
    selected_indices = []
    for label, indices in class_to_samples.items():
        k = max(1, int(len(indices) * fraction))  # at least 1 sample per class
        sampled = random.sample(indices, k)
        selected_indices.extend(sampled)
    
    # Return subset dataset
    subset = torch.utils.data.Subset(dataset, selected_indices)
    return subset


class PatchDataset(Dataset):
    def __init__(self, root, transform=None, is_valid_file=None):
        self.samples = []   # list of (filepath, label)
        self.transform = transform

        for patient_dir in sorted(os.listdir(root)):
            patient_path = os.path.join(root, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            for class_label in ['0', '1']:
                class_path = os.path.join(patient_path, class_label)
                if not os.path.isdir(class_path):
                    continue
                for fname in os.listdir(class_path):
                    fpath = os.path.join(class_path, fname)
                    if is_valid_file is not None and not is_valid_file(fpath):
                        continue
                    self.samples.append((fpath, int(class_label)))

        self.classes = ['0', '1']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img = Image.open(fpath).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_path = "/data/B3"
    full_dataset = PatchDataset(root=data_path, transform=None, is_valid_file=valid_file)
    
    class_names = full_dataset.classes
    
    data_label = "data_B3"
    
    print(f"Number of classes: {len(class_names)}")
    print(f"Total samples: {len(full_dataset)}")
    assert len(class_names) == num_classes, f"Expected {num_classes} classes, but found {len(class_names)}"
    
    # ── 1. Extract patient IDs from file paths ───────────────────────────────────
    # Filename format: {patientID}_idx{N}_x{X}_y{Y}_class{C}.png
    # The patient folder IS the patient ID (top-level subdirs of data3/0/ and data3/1/)
    # We extract it from the filename prefix instead to be robust.

    def get_patient_id(filepath):
        """Extract patient ID from patch filename. e.g. '10253_idx5_x1351_y1101_class0.png' → '10253'"""
        fname = os.path.basename(filepath)
        match = re.match(r'^(\d+)_', fname)
        if match:
            return match.group(1)
        # fallback: use the parent-parent folder name (the patient ID folder)
        return os.path.basename(os.path.dirname(os.path.dirname(filepath)))

    # Build index: patient_id → list of (dataset_index, label)
    patient_to_indices = defaultdict(list)
    for idx in range(len(full_dataset)):
        filepath, label = full_dataset.samples[idx]
        pid = get_patient_id(filepath)
        patient_to_indices[pid].append((idx, label))

    all_patient_ids = np.array(sorted(patient_to_indices.keys()))

    # Compute a per-patient label for stratification:
    # Use fraction of IDC patches (class 1) — then binarise at 0.5 for stratification.
    patient_idc_ratio = {}
    for pid, entries in patient_to_indices.items():
        labels = [lbl for _, lbl in entries]
        patient_idc_ratio[pid] = int(np.mean(labels) >= 0.5)

    patient_strat_labels = np.array([patient_idc_ratio[pid] for pid in all_patient_ids])

    logger.info(f"Total patients          : {len(all_patient_ids)}")
    logger.info(f"IDC-majority patients   : {patient_strat_labels.sum()}")
    logger.info(f"non-IDC-majority patients: {(patient_strat_labels == 0).sum()}")

    # ── 2. Patient-level train/val/test split ────────────────────────────────────
    trainval_pids, test_pids, trainval_strat, _ = train_test_split(
        all_patient_ids, patient_strat_labels,
        test_size=0.20,
        stratify=patient_strat_labels,
        random_state=42
    )

    # Expand patient IDs → patch indices
    def pids_to_patch_indices(pids):
        indices, labels = [], []
        for pid in pids:
            for idx, lbl in patient_to_indices[pid]:
                indices.append(idx)
                labels.append(lbl)
        return np.array(indices), np.array(labels)

    test_patch_indices, test_patch_labels   = pids_to_patch_indices(test_pids)
    trainval_patch_indices, trainval_patch_labels = pids_to_patch_indices(trainval_pids)

    test_subset = Subset(full_dataset, test_patch_indices.tolist())

    logger.info(f"Test patients       : {len(test_pids)}  →  {len(test_patch_indices)} patches")
    logger.info(f"Train+val patients  : {len(trainval_pids)}  →  {len(trainval_patch_indices)} patches")

    # ── 3. Patient-level StratifiedKFold ─────────────────────────────────────────
    # Fold on PATIENTS, then expand to patches — no leakage possible.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    trainval_strat_labels = np.array([patient_idc_ratio[pid] for pid in trainval_pids])

    summary_rows = []
    
    for model_name, model_fn in model_configs:
        logger.info(f"\n{'═'*60}\nModel: {model_name}\n{'═'*60}")
        fold_metrics     = []
        fold_train_times = []
        fold_infer_times = []

        for fold_idx, (rel_train_pid_idx, rel_val_pid_idx) in enumerate(
            skf.split(trainval_pids, trainval_strat_labels), start=1
        ):
            logger.info(f"\n{'─'*50}  Fold {fold_idx}/{N_FOLDS}  {'─'*50}")

            train_pids = trainval_pids[rel_train_pid_idx]
            val_pids   = trainval_pids[rel_val_pid_idx]

            # Expand to patch indices
            abs_train_idx, _ = pids_to_patch_indices(train_pids)
            abs_val_idx,   _ = pids_to_patch_indices(val_pids)

            # Sanity check: zero patient overlap across all three splits
            assert set(train_pids).isdisjoint(set(val_pids)),   "LEAKAGE: train/val patient overlap!"
            assert set(train_pids).isdisjoint(set(test_pids)),  "LEAKAGE: train/test patient overlap!"
            assert set(val_pids).isdisjoint(set(test_pids)),    "LEAKAGE: val/test patient overlap!"

            train_loader = DataLoader(
                TransformedDataset(Subset(full_dataset, abs_train_idx.tolist()), transform),
                batch_size=32, shuffle=True, num_workers=4, pin_memory=True
            )
            val_loader = DataLoader(
                TransformedDataset(Subset(full_dataset, abs_val_idx.tolist()), transform),
                batch_size=32, shuffle=False, num_workers=4, pin_memory=True
            )
            test_loader = DataLoader(
                TransformedDataset(test_subset, transform),
                batch_size=32, shuffle=False, num_workers=4, pin_memory=True
            )

            logger.info(f"  Train: {len(train_pids)} patients  →  {len(abs_train_idx)} patches")
            logger.info(f"  Val  : {len(val_pids)} patients  →  {len(abs_val_idx)} patches")
            logger.info(f"  Test : {len(test_pids)} patients  →  {len(test_patch_indices)} patches")

            # --- rest of your fold loop is unchanged from here ---
            model = model_fn(num_classes, device)
            model, train_time = train_model(
                model, train_loader, val_loader,
                device, data_label, model_name, fold_idx,
                num_epochs=num_epochs
            )
            infer_start = time.time()
            acc, f1_macro, auc_roc, f1_micro, f1_weighted, auprc, \
                y_true, y_pred, y_scores = evaluate(model, test_loader, device)
            infer_time = time.time() - infer_start

            log_model_results(
                data_label, model_name, fold_idx, class_names,
                y_true, y_pred, y_scores,
                acc, f1_macro, auc_roc, f1_micro, f1_weighted, auprc,
                train_time, infer_time
            )
            fold_metrics.append({
                "Accuracy":    acc,
                "F1-macro":    f1_macro,
                "F1-micro":    f1_micro,
                "F1-weighted": f1_weighted,
                "AUC-ROC":     auc_roc,
                "AUPRC":       auprc,
            })
            fold_train_times.append(train_time)
            fold_infer_times.append(infer_time)

        # ── Aggregate across folds ────────────────────────────────────────
        aggregated = aggregate_folds(fold_metrics)
        log_mean_sd_summary(model_name, aggregated)

        row = {"Model": model_name, "Split": f"{N_FOLDS}-fold CV"}
        for metric, (mean, sd) in aggregated.items():
            row[f"{metric}_mean"] = round(mean, 4)
            row[f"{metric}_sd"]   = round(sd,   4)
            row[f"{metric}"]      = f"{mean:.4f} ± {sd:.4f}"
        row["Train_Time_mean_sec"] = round(float(np.mean(fold_train_times)), 1)
        row["Infer_Time_mean_sec"] = round(float(np.mean(fold_infer_times)), 3)
        summary_rows.append(row)

    # ── Final summary CSV ─────────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)

    readable_cols = ["Model", "Split"] + [
        m for m in ["Accuracy", "F1-macro", "F1-micro", "F1-weighted", "AUC-ROC", "AUPRC"]
        if m in df.columns
    ]
    numeric_cols = [c for c in df.columns if c not in readable_cols]
    df = df[readable_cols + numeric_cols]

    csv_path = f"DL_{data_label}_{N_FOLDS}fold_results.csv"
    df.to_csv(csv_path, index=False)

    logger.info("\n" + "=" * 70)
    logger.info(f"FINAL CROSS-VALIDATION SUMMARY  ({N_FOLDS} folds)")
    logger.info("=" * 70)
    logger.info("\n" + df[readable_cols].to_string(index=False))
    logger.info(f"\nFull results saved to: {csv_path}")
    logger.info(f"Full log saved to: {log_filename}")


if __name__ == "__main__":
    main()