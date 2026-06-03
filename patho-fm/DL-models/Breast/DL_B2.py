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
import pandas as pd
import timm 
import logging
import time
from datetime import datetime


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
num_classes = 8
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_path = "/data/B2"
    full_dataset = datasets.ImageFolder(root=data_path, transform=None, is_valid_file=valid_file)
    
    class_names = full_dataset.classes
    
    data_label = "data_B2"
    
    print(f"Number of classes: {len(class_names)}")
    print(f"Total samples: {len(full_dataset)}")
    assert len(class_names) == num_classes, f"Expected {num_classes} classes, but found {len(class_names)}"
    
    all_labels  = np.array([full_dataset[i][1] for i in range(len(full_dataset))])
    all_indices = np.arange(len(full_dataset))
 
    # ── Hold out a fixed test set (20 %) — stratified, seed=42 ───────────
    # The remaining 80 % is used for 3-fold CV (train / val splits).
    trainval_indices, test_indices, trainval_labels, _ = train_test_split(
        all_indices, all_labels,
        test_size=0.20,
        stratify=all_labels,
        random_state=42
    )
 
    test_subset = Subset(full_dataset, test_indices.tolist())
    logger.info(f"Hold-out test set size : {len(test_subset)}")
    logger.info(f"Train+val pool size    : {len(trainval_indices)}")
    
    # ── StratifiedKFold on the 80% train+val pool ────────────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    summary_rows = []

    for model_name, model_fn in model_configs:
        logger.info(f"\n{'═'*60}\nModel: {model_name}\n{'═'*60}")

        fold_metrics     = []
        fold_train_times = []
        fold_infer_times = []

        for fold_idx, (rel_train_idx, rel_val_idx) in enumerate(
            skf.split(trainval_indices, trainval_labels), start=1
        ):
            logger.info(f"\n{'─'*50}  Fold {fold_idx}/{N_FOLDS}  {'─'*50}")

            abs_train_idx = trainval_indices[rel_train_idx].tolist()
            abs_val_idx   = trainval_indices[rel_val_idx].tolist()

            train_loader = DataLoader(
                TransformedDataset(Subset(full_dataset, abs_train_idx), transform),
                batch_size=32, shuffle=True, num_workers=4, pin_memory=True
            )
            val_loader = DataLoader(
                TransformedDataset(Subset(full_dataset, abs_val_idx), transform),
                batch_size=32, shuffle=False, num_workers=4, pin_memory=True
            )
            test_loader = DataLoader(
                TransformedDataset(test_subset, transform),
                batch_size=32, shuffle=False, num_workers=4, pin_memory=True
            )

            logger.info(f"  Train samples : {len(abs_train_idx)}")
            logger.info(f"  Val samples   : {len(abs_val_idx)}")
            logger.info(f"  Test samples  : {len(test_subset)}")

            # Fresh model for each fold — re-initialise from scratch
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