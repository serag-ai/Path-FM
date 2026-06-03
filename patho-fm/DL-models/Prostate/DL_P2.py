# Import the necessary modules

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import vit_b_16, convnext_base, resnet50, densenet121, swin_t
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    average_precision_score
)
import os
import pandas as pd
import logging
import time
from datetime import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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

# Common parameters
TARGET_SIZE = 224
RESIZE_SIZE = 256
INTERPOLATION = InterpolationMode.BILINEAR 
num_classes = 4
num_epochs = 5 
N_FOLDS    = 3   # number of cross-validation folds

# ============================================================================
# DATASET transform
# ============================================================================

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



class ExcelLabeledDataset(Dataset):
    def __init__(self, excel_path, image_folder, transform=None):
        """
        Args:
            excel_path: Path to the Excel file with 'Patch filename' and 'MV' columns
            image_folder: Path to the folder containing images
            transform: Optional torchvision transform to be applied on images
        """
        self.df = pd.read_excel(excel_path)
        self.image_folder = image_folder
        self.transform = transform
        
        # Rename columns for easier access if needed
        if 'Patch filename' in self.df.columns:
            self.df.rename(columns={'Patch filename': 'image_name'}, inplace=True)
        if 'MV' not in self.df.columns:
            raise ValueError("Excel file must contain a column named 'MV' for labels.")

        # Drop rows where MV is not numeric (e.g., 'MV', 'ground truth')
        self.df = self.df[pd.to_numeric(self.df['MV'], errors='coerce').notna()]
        self.df['MV'] = self.df['MV'].astype(int)

        unique_labels = self.df['MV'].unique()
        print(f"Unique labels found in MV column ({len(unique_labels)}): {unique_labels}")
        
        # Add .jpg extension to filenames
        self.df['image_name'] = self.df['image_name'].astype(str) + '.jpg'
        
        # Filter out rows with missing images
        valid_indices = []
        missing_count = 0
        
        for idx, img_name in enumerate(self.df['image_name']):
            img_path = os.path.join(self.image_folder, img_name)
            if os.path.exists(img_path):
                valid_indices.append(idx)
            else:
                missing_count += 1
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        
        if missing_count > 0:
            print(f" Warning: {missing_count} images not found and excluded from dataset")
        print(f" Dataset size: {len(self.df)} images")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image name and path
        img_name = self.df.iloc[idx]['image_name']
        img_path = os.path.join(self.image_folder, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Read label (integer from 0 to 3)
        label = int(self.df.iloc[idx]['MV'])
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
class TransformedDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        # If img is already a tensor, convert back to PIL for transform
        if not isinstance(img, Image.Image):
            img = self.base_dataset.dataset[self.base_dataset.indices[idx]][0]
        img = self.transform(img)
        return img, label


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    excel_path_train = "/data/P2/train.xlsx"
    excel_path_val   = "/data/P2/val.xlsx"
    excel_path_test  = "/data/P2/test.xlsx"
    image_folder     = "/data/P2/Patches"

    train_dataset = ExcelLabeledDataset(excel_path_train, image_folder, transform=None)
    val_dataset   = ExcelLabeledDataset(excel_path_val,   image_folder, transform=None)
    test_dataset  = ExcelLabeledDataset(excel_path_test,  image_folder, transform=None)

    data_label  = "data_P2"
    SEEDS       = [0, 1, 2]

    # Derive class info from the training Excel
    unique_labels = sorted(train_dataset.df['MV'].unique())
    num_classes   = len(unique_labels)
    class_names   = [str(c) for c in unique_labels]

    logger.info(f"Number of classes : {num_classes}  →  {class_names}")
    logger.info(f"Train size : {len(train_dataset)}")
    logger.info(f"Val size   : {len(val_dataset)}")
    logger.info(f"Test size  : {len(test_dataset)}")

    # Pre-wrap datasets with transform — split never changes across seeds
    train_transformed = TransformedDataset(train_dataset, transform)
    val_transformed   = TransformedDataset(val_dataset,   transform)
    test_transformed  = TransformedDataset(test_dataset,  transform)

    # Test loader is identical across all seeds and models
    test_loader = DataLoader(
        test_transformed,
        batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    summary_rows = []

    for model_name, model_fn in model_configs:
        logger.info(f"\n{'═'*60}\nModel: {model_name}\n{'═'*60}")

        run_metrics     = []
        run_train_times = []
        run_infer_times = []

        for seed in SEEDS:
            logger.info(f"\n{'─'*50}  Seed {seed}  {'─'*50}")

            # Seed everything for reproducible weight init and batch order
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            g = torch.Generator()
            g.manual_seed(seed)

            train_loader = DataLoader(
                train_transformed,
                batch_size=32, shuffle=True,
                num_workers=4, pin_memory=True,
                generator=g,
            )
            val_loader = DataLoader(
                val_transformed,
                batch_size=32, shuffle=False,
                num_workers=4, pin_memory=True,
            )

            # Fresh model — weight init controlled by the seed set above
            model = model_fn(num_classes, device)

            model, train_time = train_model(
                model, train_loader, val_loader,
                device, data_label, model_name, seed,
                num_epochs=num_epochs
            )

            infer_start = time.time()
            acc, f1_macro, auc_roc, f1_micro, f1_weighted, auprc, \
                y_true, y_pred, y_scores = evaluate(model, test_loader, device)
            infer_time = time.time() - infer_start

            log_model_results(
                data_label, model_name, seed, class_names,
                y_true, y_pred, y_scores,
                acc, f1_macro, auc_roc, f1_micro, f1_weighted, auprc,
                train_time, infer_time
            )

            run_metrics.append({
                "Accuracy":    acc,
                "F1-macro":    f1_macro,
                "F1-micro":    f1_micro,
                "F1-weighted": f1_weighted,
                "AUC-ROC":     auc_roc,
                "AUPRC":       auprc,
            })
            run_train_times.append(train_time)
            run_infer_times.append(infer_time)

        # ── Aggregate across seeds ────────────────────────────────────────────
        aggregated = aggregate_folds(run_metrics)
        log_mean_sd_summary(model_name, aggregated)

        row = {"Model": model_name, "Split": "fixed (3-seed runs)"}
        for metric, (mean, sd) in aggregated.items():
            row[f"{metric}_mean"] = round(mean, 4)
            row[f"{metric}_sd"]   = round(sd,   4)
            row[f"{metric}"]      = f"{mean:.4f} ± {sd:.4f}"
        row["Train_Time_mean_sec"] = round(float(np.mean(run_train_times)), 1)
        row["Infer_Time_mean_sec"] = round(float(np.mean(run_infer_times)), 3)
        summary_rows.append(row)

    # ── Final summary CSV ─────────────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)

    readable_cols = ["Model", "Split"] + [
        m for m in ["Accuracy", "F1-macro", "F1-micro", "F1-weighted", "AUC-ROC", "AUPRC"]
        if m in df.columns
    ]
    numeric_cols = [c for c in df.columns if c not in readable_cols]
    df = df[readable_cols + numeric_cols]

    csv_path = f"DL_{data_label}_3seed_results.csv"
    df.to_csv(csv_path, index=False)

    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY — fixed split, 3-seed stability runs")
    logger.info("=" * 70)
    logger.info("\n" + df[readable_cols].to_string(index=False))
    logger.info(f"\nFull results saved to: {csv_path}")
    logger.info(f"Full log saved to: {log_filename}")


if __name__ == "__main__":
    main()