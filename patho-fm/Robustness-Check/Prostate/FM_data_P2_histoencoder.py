# Import the necessary modules
import torch
import os
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.transforms import InterpolationMode
import numpy as np
from PIL import Image
import torch.nn.functional as F  # For softmax to compute probabilities
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import histoencoder.functional as FF
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    average_precision_score
)
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


# Import the models from load_models.py
from load_models_prostate import model_prostat_small, model_prostat_medium

batch_size = 8
num_epochs = 5
num_classes = 4

# Common parameters
TARGET_SIZE = 224
RESIZE_SIZE = 256
INTERPOLATION = InterpolationMode.BILINEAR 

# Standard ImageNet transform
transform = transforms.Compose([
    transforms.Resize(RESIZE_SIZE, interpolation=INTERPOLATION),
    transforms.CenterCrop(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
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
# FOUNDATION MODEL WRAPPER FOR PROSTAT ONLY
# ============================================================================
class ClassificationModel(nn.Module):
    def __init__(self, foundation_model, feature_dim, num_classes=num_classes, model_type='prostat'):
        super().__init__()
        self.foundation_model = foundation_model
        self.feature_dim = feature_dim
        self.model_type = model_type.lower()
        self.head = LinearClassifier(feature_dim, num_classes)
        
        # Freeze the foundation model
        for param in self.foundation_model.parameters():
            param.requires_grad = False

        print(f"\nProcessing model: {self.model_type.upper()}")
        self.get_trainable_params()

    # ------------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------------
    def forward(self, x):
        if self.model_type == 'prostat':
            features = FF.extract_features(self.foundation_model, x)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
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
            print("No trainable parameters found! Only the head is trainable.")
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

class ExcelLabeledDataset2(Dataset):
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
    
    
class ExcelLabeledDataset3(Dataset):
    def __init__(self, excel_path, image_folder, transform=None):
        """
        Args:
            excel_path: Path to the Excel file with labels
            image_folder: Path to the folder containing images
            transform: Optional transform to be applied on images
        """
        self.df = pd.read_excel(excel_path)
        self.image_folder = image_folder
        self.transform = transform
        
        # Get label columns (all columns except image_name)
        self.label_columns = [col for col in self.df.columns if col != 'image_name']
        
        # Filter out rows where images don't exist
        valid_indices = []
        missing_count = 0
        
        for idx, img_name in enumerate(self.df['image_name']):
            img_path = os.path.join(self.image_folder, img_name)
            if os.path.exists(img_path):
                valid_indices.append(idx)
            else:
                missing_count += 1
        
        # Keep only rows with existing images
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        
        if missing_count > 0:
            print(f"Warning: {missing_count} images not found and excluded from dataset")
            print(f"Dataset size: {len(self.df)} images")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image name and path
        img_name = self.df.iloc[idx]['image_name']
        img_path = os.path.join(self.image_folder, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get label (find which column has value 1)
        labels = self.df.iloc[idx][self.label_columns].values
        label = labels.argmax()  # Get index of the class with value 1
        
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
    
def valid_file(path):
    return not path.endswith(".DS_Store")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define dataset paths
    main_path = "/data"
    # Define subfolders for each dataset
    external_datasets = {
            "P1": os.path.join(main_path, "P1"),
            "P3": os.path.join(main_path, "P3")
        }

    # Load train (P2)
    excel_path = "/data/P2/annot.xlsx"  
    image_folder = "/data/P2/Patches"

     # Create the full dataset 
    train_full_dataset = ExcelLabeledDataset2(excel_path, image_folder, transform=None)
    print(len(train_full_dataset))           # Number of valid images 
    
    # Print dataset size
    df = pd.read_excel(excel_path)
    print(f"Fine-tuning Dataset size: {len(df)} images")

    # If labels are stored in the 'MV' column
    if 'MV' in df.columns:
        unique_labels = df['MV'].unique()
        num_classes = len(unique_labels)
        
        # Convert to string for classification_report
        class_names = [str(c) for c in unique_labels]
    
        print(f"Number of classes: {num_classes}")
        print(f"Class labels: {unique_labels}")
    else:
        print("Column 'MV' not found in the Excel file.")

    # Split P1 into train/val (e.g., 80/20)
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

    print(f"Train size (P2): {len(train_dataset)}")
    print(f"Val size (P2):   {len(val_dataset)}")
    
    # ----------------------------
    # Load external datasets (custom logic per dataset)
    # ----------------------------
    def load_external_dataset(name, data_path, transform):
        """
        Custom loader that adapts to dataset-specific structure or labeling.
        You can extend this for each dataset.
        """
        if name == "P1":
            full_dataset = datasets.ImageFolder(root=data_path, transform=None, is_valid_file=valid_file)
            return full_dataset
        elif name == "P3":
            excel_path = "/data/P3/annot.xlsx"  
            image_folder = "/data/P3/images"

            # Create the full dataset
            full_dataset = ExcelLabeledDataset3(excel_path, image_folder, transform=None)
            return full_dataset
        else:
            return datasets.ImageFolder(root=path, transform=transform)

    # Build dictionary for test datasets
    test_datasets_all = {}
    for name, path in external_datasets.items():
        try:
            ds = load_external_dataset(name, path, transform=None)
            print(f"Loaded {name} (size={len(ds)})")
            test_datasets_all[name] = ds
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            
    train_datasets = {
        "prostates": TransformedDataset(train_dataset, transform),
        "prostatem": TransformedDataset(train_dataset, transform),
    }
    
    val_datasets = {
        k: TransformedDataset(val_dataset, transform) for k in train_datasets.keys()
    }

    # Build test datasets dict per model
    test_datasets_per_model = {}
    for test_name, ds in test_datasets_all.items():
        test_datasets_per_model[test_name] = {
            "prostates": TransformedDataset(ds, transform),
            "prostatem": TransformedDataset(ds, transform)
        }

    # Model configurations
    model_configs = [
            ("prostates", model_prostat_small, 384, 'prostat'),
            ("prostatem", model_prostat_medium, 512, 'prostat'),
    ]
        
    data_label = "P2"  

    all_results = []

    for model_name, fm_model, feature_dim, model_type in model_configs:
        if fm_model is None:
            print(f"Skipping {model_name} because the model failed to load.")
            continue

        print(f"\n=== Fine-tuning {model_name} on P2 ===")
        
        logger.info(f"\n{'─'*60}\nProcessing model: {model_name}\n{'─'*60}")

        train_loader = DataLoader(train_datasets[model_name], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_datasets[model_name], batch_size=batch_size, shuffle=False)

        # Train model on P2
        model = ClassificationModel(fm_model.to(device), feature_dim, model_type=model_type).to(device)
        model, train_time = train_model(model, train_loader, val_loader, device, data_label, model_name, num_epochs=num_epochs)
                         
        # Test model on P1, P3
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
        csv_path = "FM_cross_P2_histo_results.csv"
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