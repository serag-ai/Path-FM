# Import the necessary modules
import warnings
warnings.simplefilter("always", DeprecationWarning)
import torch
import os
import re
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.transforms import InterpolationMode
from transformers import AutoImageProcessor
import torch.nn.functional as F  # For softmax to compute probabilities
from sklearn.model_selection import StratifiedKFold, train_test_split
import h5py
import random
from PIL import Image
from torch.utils.data import Subset, ConcatDataset
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    average_precision_score, precision_recall_curve, auc
)
import sys
from pathlib import Path
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

#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True

# Import the models from load_models.py
from load_models import model_gigapath, model_uni, model_plip, model_conch, model_phikon, model_hibou, model_pathorchestra

batch_size = 8
num_epochs = 5
num_classes = 2

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
hibou_processor = AutoImageProcessor.from_pretrained("histai/hibou-L", trust_remote_code=True, force_download=True)

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

# Phikon transform
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
        print(f"Warning: Could not load Phikon processor, using ImageNet normalization: {e}")
        return transform

transform_phikon = get_phikon_transform()

transform_hibou = transforms.Compose([
    transforms.Resize(RESIZE_SIZE, interpolation=INTERPOLATION),
    transforms.CenterCrop(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7068, 0.5755, 0.7220], std=[0.1950, 0.2316, 0.1816]),
])

# ============================================================================
# CLASSIFIER HEAD
# ============================================================================
class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes=num_classes):
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

def train_model(model, train_loader, val_loader, device, data_label, model_name, num_epochs=50):
    """
    Train
    """
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    save_dir = "Best_trained_models"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_{model_name}_{data_label}.pth")
    
    train_start = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images) 
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation
        val_acc, _, _, _, _, _, _, _, _ = evaluate(model, val_loader, device)
        epoch_secs = time.time() - epoch_start
        
        # Step scheduler
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Acc: {val_acc:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved (Val Acc: {best_val_acc:.4f})")
    
    train_elapsed = time.time() - train_start
    model.load_state_dict(torch.load(best_model_path))
    return model, train_elapsed


def evaluate(model, dataloader, device):
    """
    Returns: acc, f1_macro, auc_roc, f1_micro, f1_weighted, auprc
             plus raw arrays (y_true, y_pred, y_scores) used for reporting
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

    acc        = accuracy_score(y_true, y_pred)
    f1_macro   = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    f1_micro   = f1_score(y_true, y_pred, average='micro',    zero_division=0)
    f1_weighted= f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # AUC-ROC
    if y_scores.shape[1] == 2:
        auc_roc = roc_auc_score(y_true, y_scores[:, 1])
    else:
        auc_roc = roc_auc_score(y_true, y_scores, multi_class='ovr', average='macro')

    # AUPRC  (per-class then macro-average)
    auprc_per_class = []
    for c in range(y_scores.shape[1]):
        binary_labels = (y_true == c).astype(int)
        ap = average_precision_score(binary_labels, y_scores[:, c])
        auprc_per_class.append(ap)
    auprc = float(np.mean(auprc_per_class))

    return acc, f1_macro, auc_roc, f1_micro, f1_weighted, auprc, y_true, y_pred, y_scores


def log_model_results(data_label, test_name, model_name, class_names, y_true, y_pred, y_scores,
                      acc, f1_macro, auc_roc, f1_micro, f1_weighted, auprc,
                      train_time_secs, inference_time_secs):
    """Write confusion matrix, classification report and predictions to the log."""
    sep = "=" * 70
    logger.info(f"\n{sep}")
    logger.info(f"RESULTS FOR MODEL: {model_name}")
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

    # ── Save full predictions to CSV ───────────────────────────────────────
    pred_dir = "predictions"
    os.makedirs(pred_dir, exist_ok=True)
    pred_df = pd.DataFrame({
        "index":      range(len(y_true)),
        "true_label": [class_names[t] for t in y_true],
        "pred_label": [class_names[p] for p in y_pred],
        **{f"prob_{c}": y_scores[:, i] for i, c in enumerate(class_names)}
    })
    pred_csv = os.path.join(pred_dir, f"{data_label}_{test_name}_predictions_{model_name}.csv")
    pred_df.to_csv(pred_csv, index=False)
    logger.info(f"\nFull predictions saved to: {pred_csv}")
    logger.info(sep + "\n")

class PCamDataset(Dataset):
    def __init__(self, h5_file_x, h5_file_y, transform=None):
        super().__init__()
        self.h5_file_x = h5_file_x
        self.h5_file_y = h5_file_y
        self.transform = transform

        with h5py.File(self.h5_file_y, "r") as f:
            self._classes = np.unique(f["y"][:])
        with h5py.File(self.h5_file_x, "r") as f:
            self.length = f["x"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_file_x, "r") as f_x, h5py.File(self.h5_file_y, "r") as f_y:
            img = f_x["x"][idx]   # NumPy array (H, W, 3)
            label = f_y["y"][idx]

        # Convert to PIL image for torchvision transforms
        img = Image.fromarray(img.astype('uint8')).convert('RGB')

        # Safely convert label to tensor
        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.long()

        label = label.squeeze()
        return img, label
    
    @property
    def classes(self):
        return self._classes


def valid_file(path):
    return not path.endswith(".DS_Store")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define dataset paths
    main_path = "/data/labs/serag_AI_lab/users/chb4036/Projects/Pathology/data/breast"
    # Define subfolders for each dataset
    data_train_path = os.path.join(main_path, "data3")
    external_datasets = {
            "data2": os.path.join(main_path, "data2"),
            "data1_1": os.path.join(main_path, "data1_1"),
        }

    # Load train (data1)
    train_full_dataset = datasets.ImageFolder(root=data_train_path, transform=None, is_valid_file=valid_file)
    print(f"Training dataset (data3) classes: {len(train_full_dataset.classes)}")
    assert len(train_full_dataset.classes) == num_classes, f"Expected {num_classes} classes, but found {len(train_full_dataset.classes)}"

    # Split data1 into train/val (e.g., 80/20)
    labels = [train_full_dataset[i][1] for i in range(len(train_full_dataset))]
    indices = list(range(len(train_full_dataset)))

    train_indices, val_indices, train_labels, val_labels = train_test_split(
        indices,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(train_full_dataset, val_indices)

    print(f"Train size (data3): {len(train_dataset)}")
    print(f"Val size (data3):   {len(val_dataset)}")

    # ----------------------------
    # 2. Load external datasets 
    # ----------------------------
    def load_external_dataset(name, data_path, transform):
        """
        Custom loader that adapts to dataset-specific structure or labeling.
        You can extend this for each dataset.
        """
        if name == "data2":
            xtrain_path = f"{data_path}/camelyonpatch_level_2_split_train_x.h5"
            ytrain_path = f"{data_path}/camelyonpatch_level_2_split_train_y.h5"

            xval_path = f"{data_path}/camelyonpatch_level_2_split_valid_x.h5"
            yval_path = f"{data_path}/camelyonpatch_level_2_split_valid_y.h5"

            xtest_path = f"{data_path}/camelyonpatch_level_2_split_test_x.h5"
            ytest_path = f"{data_path}/camelyonpatch_level_2_split_test_y.h5"

            train_dataset = PCamDataset(xtrain_path, ytrain_path, transform=None)
            val_dataset = PCamDataset(xval_path, yval_path, transform=None)
            test_dataset = PCamDataset(xtest_path, ytest_path, transform=None)

            # Combined dataset
            full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
            return full_dataset
        elif name == "data1_1":
            return datasets.ImageFolder(root=data_path, transform=None, is_valid_file=valid_file) 

    # Build dictionary for test datasets
    test_datasets_all = {}
    for name, path in external_datasets.items():
        try:
            ds = load_external_dataset(name, path, transform=None)
            print(f"Loaded {name} (size={len(ds)})")
            test_datasets_all[name] = ds
        except Exception as e:
            print(f"Failed to load {name}: {e}")

    # ----------------------------
    # 3. Apply transforms (only on compatible datasets)
    # ----------------------------
    train_datasets = {
        "pathorchestra": TransformedDataset(train_dataset, transform),
        "Hibou": TransformedDataset(train_dataset, transform_hibou),
        "PhikonV2": TransformedDataset(train_dataset, transform_phikon),
        "GigaPath": TransformedDataset(train_dataset, transform),
        "Uni": TransformedDataset(train_dataset, transform),
        "PLIP": TransformedDataset(train_dataset, transform_plip),
        "CONCH": TransformedDataset(train_dataset, transform),
    }
    val_datasets = {
        k: TransformedDataset(val_dataset, transform) for k in train_datasets.keys()
    }

    # Build test datasets dict per model (only for those that loaded successfully)
    test_datasets_per_model = {}
    for test_name, ds in test_datasets_all.items():
        test_datasets_per_model[test_name] = {
            "pathorchestra": TransformedDataset(ds, transform),
            "Hibou": TransformedDataset(ds, transform_hibou),
            "PhikonV2": TransformedDataset(ds, transform_phikon),
            "GigaPath": TransformedDataset(ds, transform),
            "Uni": TransformedDataset(ds, transform),
            "PLIP": TransformedDataset(ds, transform_plip),
            "CONCH": TransformedDataset(ds, transform)
        }

    # Model configurations
    model_configs = [
        ("pathorchestra", model_pathorchestra, 1024, 'standard'),
        ("Hibou", model_hibou, 1024, 'hibou'),
        ("PhikonV2", model_phikon, 1024, 'phikon'),
        ("GigaPath", model_gigapath, 1536, 'standard'),
        ("Uni", model_uni, 1536, 'standard'),
        ("PLIP", model_plip, 512, 'plip'),
        ("CONCH", model_conch, 512, 'conch'),
    ]

    data_label = "data3"
    all_results = []

    for model_name, fm_model, feature_dim, model_type in model_configs:
        if fm_model is None:
            print(f"Skipping {model_name} because the model failed to load.")
            continue

        print(f"\n=== Fine-tuning {model_name} on data3 ===")
        
        logger.info(f"\n{'─'*60}\nProcessing model: {model_name}\n{'─'*60}")

        train_loader = DataLoader(train_datasets[model_name], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_datasets[model_name], batch_size=batch_size, shuffle=False)

        # Train model on data3
        model = ClassificationModel(fm_model.to(device), feature_dim, model_type=model_type).to(device)
        model, train_time = train_model(model, train_loader, val_loader, device, data_label, model_name, num_epochs=num_epochs)
                         
        # Test model on data1_1, 2
        for test_name, dataset_dict in test_datasets_per_model.items():
            # Handle both possible structures
            if isinstance(dataset_dict, dict):
                dataset = dataset_dict[model_name]
            else:
                dataset = dataset_dict  # already a dataset

            test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            # Evaluate on test set
            print(f"\nEvaluating {model_name} on test set...")
            infer_start = time.time()
            acc, f1_macro, auc_roc, f1_micro, f1_weighted, auprc, y_true, y_pred, y_scores =  \
                evaluate(model, test_loader, device)
            infer_time = time.time() - infer_start

            # Log detailed results (confusion matrix, report, predictions)
            class_names = train_full_dataset.classes
            log_model_results(data_label, test_name, model_name, class_names, y_true, y_pred, y_scores,
                              acc, f1_macro, auc_roc, f1_micro, f1_weighted, auprc,
                              train_time, infer_time)

            all_results.append((data_label, test_name, model_name, acc, f1_macro, auc_roc, f1_micro, f1_weighted, auprc,
                                train_time, infer_time))

        # ── Checkpoint CSV (saved after each model completes) ─────────────────
        rows = []
        for (dl, tn, mn, _acc, _f1_macro, _auc_roc, _f1_micro, _f1_weighted,
             _auprc, _train_time, _infer_time) in all_results:
            rows.append({
                "Model":                mn,
                "Train Data":           dl,
                "Test Data":            tn,
                "Accuracy":             round(_acc,          4),
                "F1-macro":             round(_f1_macro,     4),
                "F1-micro":             round(_f1_micro,     4),
                "F1-weighted":          round(_f1_weighted,  4),
                "AUC-ROC":             round(_auc_roc,      4),
                "AUPRC":                round(_auprc,        4),
                "Train_Time_sec":       round(_train_time,   1),
                "Inference_Time_sec":   round(_infer_time,   3),
            })
        df = pd.DataFrame(rows)
        csv_path = "FM_cross_data3_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"  [checkpoint] Results so far saved to: {csv_path}")

    # ── Final summary (outside the model loop) ────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info("\n" + df.to_string(index=False))
    logger.info(f"\nResults saved to: {csv_path}")
    logger.info(f"Full log saved to: {log_filename}")


if __name__ == "__main__":
    main()