# Import the necessary modules
import torch
import os
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.transforms import InterpolationMode
from transformers import AutoImageProcessor
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    average_precision_score
)
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Subset
import logging
import time
from datetime import datetime

# Import the models from load_models.py
from load_models import model_gigapath, model_uni, model_plip, model_conch, model_phikon, model_hibou, model_dinobloom, model_reddino, model_pathorchestra

batch_size = 8
num_epochs = 5
num_classes = 8
N_FOLDS    = 3   # number of cross-validation folds

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

phikon_processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
hibou_processor  = AutoImageProcessor.from_pretrained("histai/hibou-L", trust_remote_code=True, force_download=True)

# Standard ImageNet transform
transform = transforms.Compose([
    transforms.Resize(RESIZE_SIZE, interpolation=INTERPOLATION),
    transforms.CenterCrop(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# PLIP (CLIP) transform
transform_plip = transforms.Compose([
    transforms.Resize(RESIZE_SIZE, interpolation=INTERPOLATION),
    transforms.CenterCrop(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])

def get_phikon_transform():
    try:
        return transforms.Compose([
            transforms.Resize(RESIZE_SIZE, interpolation=INTERPOLATION),
            transforms.CenterCrop(TARGET_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=phikon_processor.image_mean,
                                 std=phikon_processor.image_std)
        ])
    except Exception as e:
        logger.warning(f"Could not load Phikon processor, using ImageNet normalization: {e}")
        return transform

transform_phikon = get_phikon_transform()

def get_hibou_transform():
    try:
        return transforms.Compose([
            transforms.Resize(RESIZE_SIZE, interpolation=INTERPOLATION),
            transforms.CenterCrop(TARGET_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=hibou_processor.image_mean,
                                 std=hibou_processor.image_std)
        ])
    except Exception as e:
        logger.warning(f"Could not load Hibou processor, falling back to hardcoded stats: {e}")
        return transforms.Compose([
            transforms.Resize(RESIZE_SIZE, interpolation=INTERPOLATION),
            transforms.CenterCrop(TARGET_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7068, 0.5755, 0.7220], std=[0.1950, 0.2316, 0.1816]),
        ])

transform_hibou = get_hibou_transform()

# RedDino transform
transform_reddino = transforms.Compose([
    transforms.Resize(RESIZE_SIZE, interpolation=INTERPOLATION),
    transforms.CenterCrop(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Map each model name → its transform
MODEL_TRANSFORMS = {
    "dinobloom":    transform,
    "reddino":      transform_reddino,
    "pathorchestra":transform,
    "Hibou":        transform_hibou,
    "PhikonV2":     transform_phikon,
    "GigaPath":     transform,
    "Uni":          transform,
    "PLIP":         transform_plip,
    "CONCH":        transform,
}

# ============================================================================
# CLASSIFIER HEAD
# ============================================================================
class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes=4):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)
    
# ============================================================================
# FOUNDATION MODEL WRAPPER FOR CLASSIFICATION
# ============================================================================
class ClassificationModel(nn.Module):
    def __init__(self, foundation_model, feature_dim, num_classes=num_classes, model_type='standard'):
        super().__init__()
        self.foundation_model = foundation_model
        self.feature_dim = feature_dim
        self.model_type = model_type.lower()
        self.head = LinearClassifier(feature_dim, num_classes)

        # Freeze all foundation model parameters first
        for param in self.foundation_model.parameters():
            param.requires_grad = False

        print(f"\n Processing model: {self.model_type.upper()}")

        # Unfreeze only the last relevant layer(s)
        self._unfreeze_last_layer()
        self.get_trainable_params()

    # ------------------------------------------------------------------------
    # Selective unfreezing
    # ------------------------------------------------------------------------
    def _unfreeze_last_layer(self):
        """Unfreeze only the last layer of the foundation model depending on type."""

        if self.model_type == 'plip':
            if hasattr(self.foundation_model, 'vision_model'):
                if hasattr(self.foundation_model.vision_model, 'encoder'):
                    last_layer = self.foundation_model.vision_model.encoder.layers[-1]
                    for param in last_layer.parameters():
                        param.requires_grad = True
                    print("Unfroze PLIP encoder last transformer layer.")
                if hasattr(self.foundation_model.vision_model, 'post_layernorm'):
                    for param in self.foundation_model.vision_model.post_layernorm.parameters():
                        param.requires_grad = True
                    print("Unfroze PLIP post-layernorm.")

        elif self.model_type == 'conch':
            if hasattr(self.foundation_model, 'visual'):
                if hasattr(self.foundation_model.visual, 'transformer'):
                    if hasattr(self.foundation_model.visual.transformer, 'resblocks'):
                        last_block = self.foundation_model.visual.transformer.resblocks[-1]
                        for param in last_block.parameters():
                            param.requires_grad = True
                        print("Unfroze CONCH last transformer block.")
                if hasattr(self.foundation_model.visual, 'ln_post'):
                    for param in self.foundation_model.visual.ln_post.parameters():
                        param.requires_grad = True
                    print("Unfroze CONCH layer norm (ln_post).")

        elif self.model_type in ['phikon', 'hibou']:
            if hasattr(self.foundation_model, 'encoder') and hasattr(self.foundation_model.encoder, 'layer'):
                last_layer = self.foundation_model.encoder.layer[-1]
                for param in last_layer.parameters():
                    param.requires_grad = True
                print(f"Unfroze {self.model_type.upper()} last encoder layer.")
            elif hasattr(self.foundation_model, 'layers'):
                last_layer = self.foundation_model.layers[-1]
                for param in last_layer.parameters():
                    param.requires_grad = True
                print(f"Unfroze {self.model_type.upper()} last block in .layers.")
            if hasattr(self.foundation_model, 'layernorm'):
                for param in self.foundation_model.layernorm.parameters():
                    param.requires_grad = True
                print("Unfroze layernorm.")
            if hasattr(self.foundation_model, 'pooler'):
                for param in self.foundation_model.pooler.parameters():
                    param.requires_grad = True
                print("Unfroze pooler.")

        elif self.model_type == 'gigapath':
            if hasattr(self.foundation_model, 'visual'):
                if hasattr(self.foundation_model.visual, 'transformer'):
                    last_block = self.foundation_model.visual.transformer.resblocks[-1]
                    for param in last_block.parameters():
                        param.requires_grad = True
                    print("Unfroze GIGAPATH last transformer block.")
                if hasattr(self.foundation_model.visual, 'ln_post'):
                    for param in self.foundation_model.visual.ln_post.parameters():
                        param.requires_grad = True
                    print("Unfroze GIGAPATH layer norm (ln_post).")

        elif self.model_type == 'uni':
            if hasattr(self.foundation_model, 'vision_model'):
                encoder = getattr(self.foundation_model.vision_model, 'encoder', None)
                if encoder and hasattr(encoder, 'layers'):
                    last_layer = encoder.layers[-1]
                    for param in last_layer.parameters():
                        param.requires_grad = True
                    print("Unfroze UNI last transformer layer.")
                if hasattr(self.foundation_model.vision_model, 'post_layernorm'):
                    for param in self.foundation_model.vision_model.post_layernorm.parameters():
                        param.requires_grad = True
                    print("Unfroze UNI post-layernorm.")

        elif self.model_type == 'pathorchestra':
            if hasattr(self.foundation_model, 'encoder'):
                if hasattr(self.foundation_model.encoder, 'blocks'):
                    last_block = self.foundation_model.encoder.blocks[-1]
                    for param in last_block.parameters():
                        param.requires_grad = True
                    print("Unfroze PATHORCHESTRA encoder last block.")
            elif hasattr(self.foundation_model, 'backbone'):
                if hasattr(self.foundation_model.backbone, 'blocks'):
                    last_block = self.foundation_model.backbone.blocks[-1]
                    for param in last_block.parameters():
                        param.requires_grad = True
                    print("Unfroze PATHORCHESTRA backbone last block.")
            if hasattr(self.foundation_model, 'norm'):
                for param in self.foundation_model.norm.parameters():
                    param.requires_grad = True
                print("Unfroze PATHORCHESTRA norm layer.")
        elif self.model_type in ['dinobloom', 'reddino']:
            # Assuming these models have an 'encoder' or 'transformer' attribute
            encoder = getattr(self.foundation_model, 'encoder', None)
            if encoder and hasattr(encoder, 'layers'):
                last_layer = encoder.layers[-1]
                for param in last_layer.parameters():
                    param.requires_grad = True
                print(f"Unfroze {self.model_type.upper()} last transformer layer.")
            # Optional: unfreeze final layernorm if it exists
            if hasattr(self.foundation_model, 'post_layernorm'):
                for param in self.foundation_model.post_layernorm.parameters():
                    param.requires_grad = True
                print(f"Unfroze {self.model_type.upper()} post-layernorm.")
        else:
            modules = list(self.foundation_model.named_modules())
            last_layer_names = ['layer4', 'stages', 'blocks', 'features']
            last_layer_found = False
            for name, module in reversed(modules):
                if any(layer_name in name for layer_name in last_layer_names):
                    try:
                        last_sublayer = list(module)[-1]
                        for param in last_sublayer.parameters():
                            param.requires_grad = True
                        print(f"Unfroze last CNN block ({name}).")
                        last_layer_found = True
                        break
                    except:
                        continue
            if not last_layer_found and modules:
                _, last_module = modules[-1]
                for param in last_module.parameters():
                    param.requires_grad = True
                print("Unfroze last fallback module.")

    # ------------------------------------------------------------------------
    # Forward pass per model type
    # ------------------------------------------------------------------------
    def forward(self, x):
        if self.model_type == 'plip':
            features = self.foundation_model.get_image_features(pixel_values=x)
        elif self.model_type == 'conch':
            features = self.foundation_model.encode_image(x)
        elif self.model_type == 'gigapath':
            features = self.foundation_model.encode_image(x)
        elif self.model_type == 'uni':
            features = self.foundation_model.get_image_features(pixel_values=x)
        elif self.model_type in ['phikon', 'hibou']:
            outputs = self.foundation_model(x)
            features = outputs.last_hidden_state[:, 0, :]
        elif self.model_type in ['dinobloom', 'reddino']:
            outputs = self.foundation_model(x)
            if hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state[:, 0, :]
            else:
                features = outputs
        elif self.model_type == 'pathorchestra':
            outputs = self.foundation_model(x)
            if hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state[:, 0, :]
            else:
                features = outputs
        else:
            features = self.foundation_model(x)
        return self.head(features)

    # ------------------------------------------------------------------------
    # Utility: show trainable parameters
    # ------------------------------------------------------------------------
    def get_trainable_params(self):
        trainable_params = []
        total_params = 0
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params.append((name, param.numel()))

        trainable_count = sum(c for _, c in trainable_params)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_count:,}")
        if trainable_count == 0:
            print("No trainable parameters found! Check unfreeze logic.")
        else:
            print("Trainable layers:")
            for name, count in trainable_params:
                print(f"  - {name}: {count:,} parameters")

        return trainable_params
    

# ── Training ───────────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, device,
                data_label, model_name, fold_idx,
                num_epochs=num_epochs):
    """
    Train linear head only.
    Checkpoint is saved when *validation macro-AUC* improves
    Returns: (best_model, training_elapsed_seconds)
    """
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-4)
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_path = "/data/H2"
    
    data_label   = "data_H2"
    
    def is_valid_file(path):
        filename = os.path.basename(path)
        return not filename.startswith('.')  # skip hidden files

    full_dataset = datasets.ImageFolder(root=data_path, transform=None, is_valid_file=is_valid_file)

    class_names  = full_dataset.classes
 
    print(f"Number of classes: {len(class_names)}")
    assert len(class_names) == num_classes, (
        f"Expected {num_classes} classes, but found {len(class_names)}"
    )
    
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
 
    model_configs = [
        ("dinobloom",     model_dinobloom,     768,  'standard'),
        ("reddino",       model_reddino,       768,  'standard'),
        ("pathorchestra", model_pathorchestra, 1024, 'standard'),
        ("Hibou",         model_hibou,         1024, 'hibou'),
        ("PhikonV2",      model_phikon,        1024, 'phikon'),
        ("GigaPath",      model_gigapath,      1536, 'standard'),
        ("Uni",           model_uni,           1536, 'standard'),
        ("PLIP",          model_plip,          512,  'plip'),
        ("CONCH",         model_conch,         512,  'conch'),
    ]
 
    # ── StratifiedKFold on the 80 % train+val pool ────────────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
 
    # Collect cross-validated mean ± SD per model for the final CSV
    summary_rows = []
 
    for model_name, fm_model, feature_dim, model_type in model_configs:
        if fm_model is None:
            logger.warning(f"Skipping {model_name} — model failed to load.")
            continue
 
        logger.info(f"\n{'═'*60}\nModel: {model_name}\n{'═'*60}")
 
        fold_metrics     = []   # list of per-fold metric dicts
        fold_train_times = []
        fold_infer_times = []
 
        for fold_idx, (rel_train_idx, rel_val_idx) in enumerate(
            skf.split(trainval_indices, trainval_labels), start=1
        ):
            logger.info(f"\n{'─'*50}  Fold {fold_idx}/{N_FOLDS}  {'─'*50}")
 
            # Map relative indices back to absolute dataset indices
            abs_train_idx = trainval_indices[rel_train_idx].tolist()
            abs_val_idx   = trainval_indices[rel_val_idx].tolist()
 
            # Build Subsets and wrap with model-specific transform
            model_transform = MODEL_TRANSFORMS[model_name]
 
            train_loader = DataLoader(
                TransformedDataset(Subset(full_dataset, abs_train_idx), model_transform),
                batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                TransformedDataset(Subset(full_dataset, abs_val_idx), model_transform),
                batch_size=batch_size, shuffle=False
            )
            test_loader = DataLoader(
                TransformedDataset(test_subset, model_transform),
                batch_size=batch_size, shuffle=False
            )
 
            logger.info(f"  Train samples : {len(abs_train_idx)}")
            logger.info(f"  Val samples   : {len(abs_val_idx)}")
            logger.info(f"  Test samples  : {len(test_subset)}")
 
            # Fresh model for each fold
            model = ClassificationModel(
                fm_model.to(device), feature_dim, model_type=model_type
            ).to(device)
 
            model, train_time = train_model(
                model, train_loader, val_loader,
                device, data_label, model_name, fold_idx
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
 
        # ── Aggregate across folds ─────────────────────────────────────────
        aggregated = aggregate_folds(fold_metrics)
        log_mean_sd_summary(model_name, aggregated)
 
        # Build one summary row: Model | metric_mean | metric_sd | ...
        row = {"Model": model_name, "Split": f"{N_FOLDS}-fold CV"}
        for metric, (mean, sd) in aggregated.items():
            row[f"{metric}_mean"] = round(mean, 4)
            row[f"{metric}_sd"]   = round(sd,   4)
            row[f"{metric}"]      = f"{mean:.4f} ± {sd:.4f}"   # human-readable column
        row["Train_Time_mean_sec"] = round(float(np.mean(fold_train_times)), 1)
        row["Infer_Time_mean_sec"] = round(float(np.mean(fold_infer_times)), 3)
        summary_rows.append(row)
 
    # ── Final summary CSV ─────────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
 
    # Put human-readable "mean ± SD" columns first for easy reading
    readable_cols = ["Model", "Split"] + [
        m for m in ["Accuracy", "F1-macro", "F1-micro", "F1-weighted", "AUC-ROC", "AUPRC"]
        if m in df.columns
    ]
    numeric_cols = [c for c in df.columns if c not in readable_cols]
    df = df[readable_cols + numeric_cols]
 
    csv_path = f"FM_{data_label}_{N_FOLDS}fold_results.csv"
    df.to_csv(csv_path, index=False)
 
    logger.info("\n" + "=" * 70)
    logger.info(f"FINAL CROSS-VALIDATION SUMMARY  ({N_FOLDS} folds)")
    logger.info("=" * 70)
    logger.info("\n" + df[readable_cols].to_string(index=False))
    logger.info(f"\nFull results (with per-metric mean & SD) saved to: {csv_path}")
    logger.info(f"Full log saved to: {log_filename}")
 


if __name__ == "__main__":
    main()