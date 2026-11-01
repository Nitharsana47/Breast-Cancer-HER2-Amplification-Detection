# MobileNetV3-Large HER2-SISH classification, full training pipeline.
# Model: mobilenet_v3_large_lastblock
# Uses the pre-split train_aug / test folders. No internal re-split.
import os
import sys
import csv
import random
import datetime
import collections

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


class Logger:
    """Redirects sys.stdout to both console and a log file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# Config

# Two pre-split directories. The train/test split is performed externally by
# data_prep/split_her2.py BEFORE augmentation. Augmentation is applied only to
# the training partition by data_prep/augment_her2_train.py. This script must
# never re-split or re-shuffle images across train and test.
TRAIN_DIR = r"C:\ML_HER2_FINAL\split\train_aug"   # augmented train pool, K-fold CV happens here
TEST_DIR  = r"C:\ML_HER2_FINAL\split\test"        # untouched originals, final eval only

IMG_SIZE      = 224
BATCH_SIZE    = 16
EPOCHS        = 30
NUM_CLASSES   = 3
K_FOLDS       = 5
MODEL_NAME    = "mobilenet_v3_large_lastblock"
LEARNING_RATE = 3e-5
RANDOM_STATE  = 42

# Must match the sub-folder names inside train_aug/ and test/
CLASS_NAMES = ["Amplified", "Non-Amplified", "Normal"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class weighting for the loss.
#   False -> plain categorical cross-entropy, every sample counted equally.
#   True  -> categorical cross-entropy with inverse-frequency class weights,
#            which compensates for Normal being roughly 2.4x the size of
#            either amplification class.
# Both are "categorical cross-entropy"; this only sets the per-class scaling.
# Flip it to match whichever the reported runs used, and make Table 4 agree.
USE_CLASS_WEIGHTS = False

# Pin cuDNN to deterministic kernels so repeated runs match exactly.
# Set False to let cuDNN autotune, which is faster but not bit-reproducible.
DETERMINISTIC = True


def set_seed(seed=RANDOM_STATE):
    """
    Seed every random source the run touches: Python, NumPy and torch on both
    CPU and GPU. Without this, head initialisation and DataLoader shuffle order
    differ on every run and no result is exactly reproducible.

    DETERMINISTIC also pins cuDNN to deterministic kernels. That costs some
    throughput but removes the last source of run-to-run drift. Set it to False
    if you need the speed and can accept small variation between runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def print_device_info():
    """Print interpreter, torch build, and GPU details."""
    print(f"  PYTHON: {sys.executable}")
    print(f"  TORCH : {torch.__version__} / cuda={torch.version.cuda}")
    print(f"\n{'='*60}")
    print(f"  DEVICE: {DEVICE}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU   : {gpu_name}")
        print(f"  VRAM  : {total_mem:.2f} GB")
    print(f"{'='*60}\n")


# Folder setup

def setup_folders(model_name):
    """Create checkpoint and result directories. Return their paths."""
    checkpoint_dir = os.path.join("checkpoints", model_name)
    results_dir    = os.path.join("results",     model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir,    exist_ok=True)
    print(f"[SETUP] Checkpoint dir : {checkpoint_dir}")
    print(f"[SETUP] Results dir    : {results_dir}")
    return checkpoint_dir, results_dir


# Dataset

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')


class CustomDataset(Dataset):
    """
    Loads images from a flat class-folder structure:
        <root>/
            Amplified/
            Non-Amplified/
            Normal/
    Returns (tensor, label_int) pairs.
    On load error: returns a black tensor so training never crashes.
    """
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"[WARN] Cannot load image {path}: {e}. Using black tensor.")
            img = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

        if self.transform:
            img = self.transform(img)
        return img, label


def collect_file_paths(data_dir, class_names):
    """
    Walk each class folder and gather (path, label_int) pairs.
    Returns two parallel lists: all_paths, all_labels.
    """
    all_paths = []
    all_labels = []
    for label_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_folder):
            print(f"[WARN] Class folder not found: {class_folder}")
            continue
        for fname in sorted(os.listdir(class_folder)):
            if fname.lower().endswith(VALID_EXTENSIONS):
                all_paths.append(os.path.join(class_folder, fname))
                all_labels.append(label_idx)
    return all_paths, all_labels


# Variant suffixes written by data_prep/augment_her2_train.py. A train_aug
# filename is "<source stem>_<variant><ext>", so stripping the suffix recovers
# the source ROI that the image was derived from.
AUG_VARIANT_SUFFIXES = ('_orig', '_rot90', '_rot180', '_rot270')


def source_roi_id(path):
    """
    Recover the source ROI identifier for one augmented training image.
    All four rotations of a ROI return the same id, which is what keeps them
    together inside a cross-validation fold.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    for suffix in AUG_VARIANT_SUFFIXES:
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def build_groups(file_paths):
    """
    Build the group vector for StratifiedGroupKFold and sanity-check it.

    If no filename carries a variant suffix, every image becomes its own group
    and grouped CV silently degrades to plain k-fold, which is exactly the
    leakage this is meant to prevent. That case is reported loudly rather than
    passing unnoticed.
    """
    groups = [source_roi_id(p) for p in file_paths]
    n_groups = len(set(groups))
    print(f"[CV] {len(file_paths)} images in {n_groups} source-ROI groups "
          f"({len(file_paths) / max(n_groups, 1):.2f} images per group)")
    if n_groups == len(file_paths):
        print("[CV][WARN] Every image is its own group, so grouped k-fold is "
              "equivalent to plain k-fold here. Check that TRAIN_DIR holds "
              "files named by augment_her2_train.py; otherwise rotations of "
              "one ROI can land in both the train and validation folds.")
    return groups


def compute_class_weights(labels):
    """
    Class weights for the loss, or None when USE_CLASS_WEIGHTS is False.

    When enabled, the balanced convention is used:
        w_c = N / (K * n_c)
    so rare classes are up-weighted and the loss is not dominated by Normal.
    A class absent from a split gets weight 0 rather than an infinity.
    """
    if not USE_CLASS_WEIGHTS:
        return None
    counts = np.bincount(np.asarray(labels, dtype=np.int64),
                         minlength=NUM_CLASSES).astype(np.float64)
    total = counts.sum()
    weights = np.zeros_like(counts)
    nonzero = counts > 0
    weights[nonzero] = total / (NUM_CLASSES * counts[nonzero])
    pretty = ", ".join(f"{CLASS_NAMES[i]}={weights[i]:.4f}"
                       for i in range(NUM_CLASSES))
    print(f"  [LOSS] Inverse-frequency class weights: {pretty}")
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def print_distribution(name, labels, class_names):
    """Print per-class counts for a loaded split."""
    counter = collections.Counter(labels)
    print(f"\n[DATA] {name} distribution ({len(labels)} samples):")
    for idx, cname in enumerate(class_names):
        print(f"        {cname}: {counter.get(idx, 0)}")


# Transforms
# No online augmentation. TRAIN_DIR already contains the offline-augmented
# training images; TEST_DIR contains untouched originals. Both share the same
# resize and normalise pipeline.

def get_transforms():
    """Resize to IMG_SIZE square, to tensor, ImageNet mean/std normalisation."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])



# Model builder

def build_model(num_classes=NUM_CLASSES):
    """
    Load MobileNetV3-Large with ImageNet weights.
    1. Freeze ALL parameters.
    2. Unfreeze the final feature block, features[-1].
    3. Replace the last classifier Linear with a fresh head for num_classes.
    4. Move to DEVICE.

    MobileNetV3-Large features has 17 entries (index 0-16); features[16] is the
    final Conv2dNormActivation producing 960 channels. classifier is
    Sequential(Linear(960,1280), Hardswish, Dropout, Linear(1280,1000)); only
    the last Linear is replaced.
    """
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    model   = models.mobilenet_v3_large(weights=weights)

    # Step 1, freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Step 2, unfreeze the final feature block
    for param in model.features[-1].parameters():
        param.requires_grad = True

    # Step 3, fresh classification head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] MobileNetV3-Large, trainable params: {trainable:,} / {total:,}")
    return model



# Checkpoint management
#
# File naming:
#   checkpoints/{MODEL_NAME}/{MODEL_NAME}_fold_{N}.pth       latest per fold
#   checkpoints/{MODEL_NAME}/{MODEL_NAME}_fold_{N}_best.pth  best val_f1 per fold
#   checkpoints/{MODEL_NAME}/{MODEL_NAME}_fold_0.pth         Phase 2 resume
#   results/{MODEL_NAME}/{MODEL_NAME}_final_model.pth        Phase 2 final
#
# Each .pth dict contains:
#   epoch        int   epoch index (0-based) when saved
#   model_state  dict  model.state_dict()
#   optim_state  dict  optimizer.state_dict()
#   timestamp    str   "%Y%m%d_%H%M%S"
#   extra_info   dict  e.g. {'val_f1': 0.91}

def save_checkpoint(model, optimizer, epoch, fold, checkpoint_dir,
                    is_best=False, extra_info=None):
    """Save model and optimizer state. Optionally write a _best copy."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    state = {
        'epoch'       : epoch,
        'model_state' : model.state_dict(),
        'optim_state' : optimizer.state_dict(),
        'timestamp'   : timestamp,
        'extra_info'  : extra_info or {}
    }
    base_path = os.path.join(checkpoint_dir, f"{MODEL_NAME}_fold_{fold}.pth")
    torch.save(state, base_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir,
                                 f"{MODEL_NAME}_fold_{fold}_best.pth")
        torch.save(state, best_path)
        print(f"  [CKPT] Best checkpoint saved -> {best_path}")
    return base_path


def load_checkpoint(model, optimizer, fold, checkpoint_dir):
    """
    Resume mid-fold from the last saved checkpoint if one exists.
    Returns start_epoch (0 if no checkpoint found).
    """
    ckpt_path = os.path.join(checkpoint_dir, f"{MODEL_NAME}_fold_{fold}.pth")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optim_state'])
        start_epoch = state['epoch'] + 1
        print(f"  [CKPT] Resumed fold {fold} from epoch {state['epoch']}")
        return start_epoch
    return 0


def load_best_checkpoint(model, fold, checkpoint_dir):
    """Load the best val_f1 checkpoint for fold-level evaluation."""
    best_path = os.path.join(checkpoint_dir,
                             f"{MODEL_NAME}_fold_{fold}_best.pth")
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(state['model_state'])
        print(f"  [CKPT] Loaded best checkpoint: {best_path}")
        return True
    return False


def find_resume_fold(fold_csv, k_folds):
    """
    On restart, determine which fold to resume from. A fold counts as complete
    only when its summary row exists in fold_results.csv, so a fold interrupted
    mid-training is resumed rather than silently skipped.
    """
    done = set()
    if os.path.exists(fold_csv):
        with open(fold_csv, 'r', newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                try:
                    done.add(int(float(row['fold'])))
                except (KeyError, ValueError):
                    continue
    for fold_idx in range(1, k_folds + 1):
        if fold_idx not in done:
            return fold_idx
    return k_folds + 1


# Train one epoch

def train_one_epoch(model, loader, criterion, optimizer, epoch):
    """
    One full training epoch.
    Prints batch progress every 10 batches.
    Returns: avg_loss, accuracy, precision, recall, f1 (macro).
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(f"    Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} "
                  f"| Loss: {loss.item():.4f}")

    avg_loss  = running_loss / len(loader.dataset)
    accuracy  = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro',
                                zero_division=0)
    recall    = recall_score(all_labels, all_preds, average='macro',
                             zero_division=0)
    f1        = f1_score(all_labels, all_preds, average='macro',
                         zero_division=0)
    return avg_loss, accuracy, precision, recall, f1


# Evaluation

def evaluate(model, loader, criterion):
    """
    Full evaluation pass.
    Returns: avg_loss, accuracy, precision, recall, f1,
             all_preds, all_labels, all_probs (softmax),
             confusion_matrix, classification_report_str.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss  = running_loss / len(loader.dataset)
    accuracy  = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro',
                                zero_division=0)
    recall    = recall_score(all_labels, all_preds, average='macro',
                             zero_division=0)
    f1        = f1_score(all_labels, all_preds, average='macro',
                         zero_division=0)
    cm        = confusion_matrix(all_labels, all_preds)
    cr        = classification_report(all_labels, all_preds,
                                      target_names=CLASS_NAMES,
                                      zero_division=0)
    all_probs = np.array(all_probs)

    return (avg_loss, accuracy, precision, recall, f1,
            all_preds, all_labels, all_probs, cm, cr)


# K-fold cross-validation, on train_aug only

def run_kfold(X_train, y_train, groups, checkpoint_dir, results_dir):
    """
    StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) over the
    AUGMENTED TRAIN SET ONLY, grouped by source ROI so that all four rotations
    of an image stay inside the same fold. The test set is never seen here.
    Each fold: fresh model, optimizer, criterion.
    Early stopping patience=5 on validation F1.
    Writes epoch_results.csv live, fold_results.csv per fold.
    Resumes from the first incomplete fold if the run was stopped midway.
    Returns fold_results, all_fold_train_losses, best_hparams.
    """
    skf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE)

    epoch_csv = os.path.join(results_dir, "epoch_results.csv")
    fold_csv  = os.path.join(results_dir, "fold_results.csv")

    epoch_fieldnames = [
        'fold', 'epoch',
        'train_loss', 'train_acc', 'train_prec', 'train_rec', 'train_f1',
        'val_loss', 'val_acc', 'val_prec', 'val_rec', 'val_f1'
    ]
    fold_fieldnames = [
        'fold', 'best_epoch', 'best_val_f1',
        'val_acc', 'val_prec', 'val_rec', 'val_f1', 'val_loss'
    ]

    # Write headers only when starting fresh, so a restart appends
    if not os.path.exists(epoch_csv):
        with open(epoch_csv, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=epoch_fieldnames).writeheader()
    if not os.path.exists(fold_csv):
        with open(fold_csv, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fold_fieldnames).writeheader()

    # Reload previously completed fold results
    fold_results = []
    if os.path.exists(fold_csv):
        with open(fold_csv, 'r', newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                row = {k: (int(float(v)) if k in ('fold', 'best_epoch')
                           else float(v))
                       for k, v in row.items()}
                fold_results.append(row)

    all_fold_train_losses = {}

    X_arr = np.array(X_train)
    y_arr = np.array(y_train)
    g_arr = np.array(groups)

    transform = get_transforms()
    early_stop_patience = 5
    resume_from_fold = find_resume_fold(fold_csv, K_FOLDS)

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(X_arr, y_arr, groups=g_arr), start=1):

        if fold_idx < resume_from_fold:
            print(f"[RESUME] Fold {fold_idx} already completed, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  K-FOLD  Fold {fold_idx}/{K_FOLDS} "
              f"(validation comes from the TRAIN split only)")
        print(f"{'='*60}")

        fold_train_ds = CustomDataset(X_arr[train_idx].tolist(),
                                      y_arr[train_idx].tolist(),
                                      transform=transform)
        fold_val_ds   = CustomDataset(X_arr[val_idx].tolist(),
                                      y_arr[val_idx].tolist(),
                                      transform=transform)

        train_loader = DataLoader(fold_train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=0,
                                  pin_memory=False)
        val_loader   = DataLoader(fold_val_ds, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0,
                                  pin_memory=False)

        # Re-seed per fold so each fold is reproducible on its own, whether
        # the run started here or resumed into this fold.
        set_seed(RANDOM_STATE + fold_idx)

        # Confirm the fold really is group-disjoint before training on it
        overlap = set(g_arr[train_idx]) & set(g_arr[val_idx])
        if overlap:
            raise AssertionError(
                f"Fold {fold_idx}: {len(overlap)} source ROI(s) appear in both "
                f"the train and validation split. Grouped CV is not holding; "
                f"do not trust these validation numbers.")
        print(f"  Groups: {len(set(g_arr[train_idx]))} train, "
              f"{len(set(g_arr[val_idx]))} val, 0 shared")

        model = build_model()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE
        )
        criterion = nn.CrossEntropyLoss(
            weight=compute_class_weights(y_arr[train_idx]))

        start_epoch = load_checkpoint(model, optimizer, fold_idx,
                                      checkpoint_dir)

        best_val_f1 = -1.0
        best_epoch = 0
        patience_count = 0
        fold_train_losses = []

        for epoch in range(start_epoch, EPOCHS):
            print(f"\n  [Fold {fold_idx}] Epoch {epoch+1}/{EPOCHS}")

            tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, epoch+1)

            val_loss, val_acc, val_prec, val_rec, val_f1, \
                _, _, _, _, _ = evaluate(model, val_loader, criterion)

            fold_train_losses.append(tr_loss)

            print(f"  Train -> loss:{tr_loss:.4f} acc:{tr_acc:.4f} "
                  f"prec:{tr_prec:.4f} rec:{tr_rec:.4f} f1:{tr_f1:.4f}")
            print(f"  Val   -> loss:{val_loss:.4f} acc:{val_acc:.4f} "
                  f"prec:{val_prec:.4f} rec:{val_rec:.4f} f1:{val_f1:.4f}")

            row = {
                'fold': fold_idx, 'epoch': epoch+1,
                'train_loss': round(tr_loss,  5),
                'train_acc' : round(tr_acc,   5),
                'train_prec': round(tr_prec,  5),
                'train_rec' : round(tr_rec,   5),
                'train_f1'  : round(tr_f1,    5),
                'val_loss'  : round(val_loss, 5),
                'val_acc'   : round(val_acc,  5),
                'val_prec'  : round(val_prec, 5),
                'val_rec'   : round(val_rec,  5),
                'val_f1'    : round(val_f1,   5),
            }
            with open(epoch_csv, 'a', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=epoch_fieldnames).writerow(row)

            is_best = val_f1 > best_val_f1
            save_checkpoint(model, optimizer, epoch, fold_idx,
                            checkpoint_dir, is_best=is_best,
                            extra_info={'val_f1': val_f1})

            if is_best:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= early_stop_patience:
                    print(f"  [EARLY STOP] No improvement for "
                          f"{early_stop_patience} epochs. Stopping fold.")
                    break

        loaded = load_best_checkpoint(model, fold_idx, checkpoint_dir)
        if not loaded:
            print("  [WARN] Best checkpoint not found, using last weights.")

        val_loss, val_acc, val_prec, val_rec, val_f1, \
            _, _, _, _, _ = evaluate(model, val_loader, criterion)

        print(f"\n  [Fold {fold_idx}] BEST Val -> "
              f"acc:{val_acc:.4f} f1:{val_f1:.4f} (epoch {best_epoch})")

        fold_row = {
            'fold'       : fold_idx,
            'best_epoch' : best_epoch,
            'best_val_f1': round(best_val_f1, 5),
            'val_acc'    : round(val_acc,  5),
            'val_prec'   : round(val_prec, 5),
            'val_rec'    : round(val_rec,  5),
            'val_f1'     : round(val_f1,   5),
            'val_loss'   : round(val_loss, 5),
        }
        with open(fold_csv, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fold_fieldnames).writerow(fold_row)

        fold_results.append(fold_row)
        all_fold_train_losses[fold_idx] = fold_train_losses

    print(f"\n{'='*60}")
    print("  K-FOLD SUMMARY (validation results, from the TRAIN split only)")
    print(f"{'='*60}")
    accs  = [r['val_acc'] for r in fold_results]
    f1s   = [r['val_f1'] for r in fold_results]
    precs = [r['val_prec'] for r in fold_results]
    recs  = [r['val_rec'] for r in fold_results]
    print(f"  Val Acc  : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  Val F1   : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"  Val Prec : {np.mean(precs):.4f} +/- {np.std(precs):.4f}")
    print(f"  Val Rec  : {np.mean(recs):.4f} +/- {np.std(recs):.4f}")

    best_fold_row = max(fold_results, key=lambda r: r['val_f1'])
    best_hparams = {
        'learning_rate': LEARNING_RATE,
        'best_fold'    : best_fold_row['fold'],
        'best_epochs'  : best_fold_row['best_epoch'],
    }
    print(f"\n  Best fold: {best_fold_row['fold']} "
          f"(val_f1={best_fold_row['val_f1']:.4f})")

    return fold_results, all_fold_train_losses, best_hparams


# Phase 2: retrain on the full train_aug set

def retrain_on_full_train(X_train, y_train, best_hparams,
                          checkpoint_dir, results_dir):
    """
    Retrain a fresh model on the ENTIRE train_aug split.
    Uses the learning rate carried over from K-fold best_hparams.
    Early stopping patience=5 on TRAINING F1. Training F1 almost always keeps
    improving, so in practice this runs the full EPOCHS; there is no held-out
    set in this phase. Describe it as a fixed-epoch retrain when writing up.
    Resumes automatically if a Phase 2 checkpoint exists.
    Saves combined_training.csv and the final model .pth.
    """
    print(f"\n{'='*60}")
    print("  PHASE 2  RETRAIN ON FULL TRAIN_AUG SET")
    print(f"{'='*60}")

    final_model_path = os.path.join(results_dir,
                                    f"{MODEL_NAME}_final_model.pth")

    lr = best_hparams.get('learning_rate', LEARNING_RATE)

    transform = get_transforms()
    train_ds = CustomDataset(X_train, y_train, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0,
                              pin_memory=False)

    set_seed(RANDOM_STATE)

    model = build_model()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(y_train))

    combined_csv = os.path.join(results_dir, "combined_training.csv")
    fieldnames = ['epoch', 'train_loss', 'train_acc',
                  'train_prec', 'train_rec', 'train_f1']

    start_epoch = 0
    train_history = []

    # Phase 2 resume uses fold slot 0
    phase2_ckpt = os.path.join(checkpoint_dir, f"{MODEL_NAME}_fold_0.pth")
    if os.path.exists(phase2_ckpt):
        state = torch.load(phase2_ckpt, map_location=DEVICE)
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optim_state'])
        start_epoch = state['epoch'] + 1
        print(f"  [CKPT] Resumed Phase 2 from epoch {state['epoch']}")
        if os.path.exists(combined_csv):
            with open(combined_csv, 'r', newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    train_history.append(
                        {k: (int(v) if k == 'epoch' else float(v))
                         for k, v in row.items()})

    if not os.path.exists(combined_csv):
        with open(combined_csv, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    best_f1 = max([r['train_f1'] for r in train_history], default=-1.0)
    patience_count = 0
    early_patience = 5
    epoch = start_epoch - 1   # guard if the loop below never runs

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n  [Phase2] Epoch {epoch+1}/{EPOCHS}")
        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch+1)

        print(f"  Train -> loss:{tr_loss:.4f} acc:{tr_acc:.4f} "
              f"prec:{tr_prec:.4f} rec:{tr_rec:.4f} f1:{tr_f1:.4f}")

        row = {
            'epoch'      : epoch+1,
            'train_loss' : round(tr_loss,  5),
            'train_acc'  : round(tr_acc,   5),
            'train_prec' : round(tr_prec,  5),
            'train_rec'  : round(tr_rec,   5),
            'train_f1'   : round(tr_f1,    5),
        }
        with open(combined_csv, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
        train_history.append(row)

        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
        }, phase2_ckpt)

        if tr_f1 > best_f1:
            best_f1 = tr_f1
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= early_patience:
                print(f"  [EARLY STOP] Phase 2, no improvement for "
                      f"{early_patience} epochs.")
                break

    torch.save({
        'model_state': model.state_dict(),
        'epoch'      : epoch + 1,
        'hparams'    : best_hparams,
    }, final_model_path)
    print(f"\n  [Phase2] Final model saved -> {final_model_path}")

    return model, train_history


# Phase 3: test set evaluation, originals only

def evaluate_on_test(model, X_test, y_test, results_dir, train_history):
    """
    Evaluate the final model on the held-out TEST set: original images,
    never augmented, never used in K-fold validation or training.
    Saves test_results.csv with last-epoch train metrics alongside test metrics.
    """
    print(f"\n{'='*60}")
    print("  PHASE 3  TEST SET EVALUATION (held-out originals)")
    print(f"{'='*60}")

    test_transform = get_transforms()
    test_ds = CustomDataset(X_test, y_test, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0,
                             pin_memory=False)
    # Unweighted here: test loss should reflect the true test distribution.
    # Accuracy, macro precision/recall/F1 are unaffected by this choice.
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, test_prec, test_rec, test_f1, \
        test_preds, test_labels, test_probs, test_cm, test_cr = \
        evaluate(model, test_loader, criterion)

    print(f"  Test -> loss:{test_loss:.4f} acc:{test_acc:.4f} "
          f"prec:{test_prec:.4f} rec:{test_rec:.4f} f1:{test_f1:.4f}")
    print(f"\n  Classification Report:\n{test_cr}")

    last_train_loss = train_history[-1]['train_loss'] if train_history else None
    last_train_acc  = train_history[-1]['train_acc']  if train_history else None
    last_train_prec = train_history[-1]['train_prec'] if train_history else None
    last_train_rec  = train_history[-1]['train_rec']  if train_history else None
    last_train_f1   = train_history[-1]['train_f1']   if train_history else None

    test_csv = os.path.join(results_dir, "test_results.csv")
    with open(test_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['train_loss_last_epoch',      last_train_loss])
        writer.writerow(['train_accuracy_last_epoch',  last_train_acc])
        writer.writerow(['train_precision_last_epoch', last_train_prec])
        writer.writerow(['train_recall_last_epoch',    last_train_rec])
        writer.writerow(['train_f1_last_epoch',        last_train_f1])
        writer.writerow(['test_loss',      round(test_loss, 5)])
        writer.writerow(['test_accuracy',  round(test_acc,  5)])
        writer.writerow(['test_precision', round(test_prec, 5)])
        writer.writerow(['test_recall',    round(test_rec,  5)])
        writer.writerow(['test_f1',        round(test_f1,   5)])
    print(f"  [SAVE] test_results -> {test_csv}")

    return (test_loss, test_acc, test_prec, test_rec, test_f1,
            test_preds, test_labels, test_probs, test_cm, test_cr)


# Visualisations, 300 DPI, saved to results_dir

def save_visualisations(fold_results, all_fold_train_losses,
                        train_history,
                        test_labels, test_preds, test_probs,
                        test_cm, results_dir):
    """
    (a) plot_a_fold_loss_curves.png      train loss per epoch, all folds
    (b) plot_b_train_test_accuracy.png   Phase 2 train accuracy per epoch
    (c) plot_c_confusion_matrix.png      confusion matrix heatmap, test
    (d) plot_d_roc_curves.png            ROC one-vs-rest per class, test
    (e) plot_e_precision_recall.png      precision-recall per class, test
    (g) plot_g_auc_bar.png               per-class AUC bar chart, test
    All figures are closed after saving to free memory.
    """
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(10, 6))
    for fold_idx, losses in all_fold_train_losses.items():
        ax.plot(range(1, len(losses)+1), losses, label=f"Fold {fold_idx}")
    ax.set_title("Train Loss per Epoch, All Folds")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    path_a = os.path.join(results_dir, "plot_a_fold_loss_curves.png")
    plt.savefig(path_a, dpi=300)
    plt.close(fig)
    print(f"  [PLOT] (a) Saved -> {path_a}")

    train_accs   = [r['train_acc'] for r in train_history]
    train_epochs = [r['epoch']     for r in train_history]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_epochs, train_accs,
            label="Train Accuracy (Phase 2)", marker='o', markersize=3)
    ax.set_title("Train Accuracy per Epoch (Phase 2, Full Train Set)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    path_b = os.path.join(results_dir, "plot_b_train_test_accuracy.png")
    plt.savefig(path_b, dpi=300)
    plt.close(fig)
    print(f"  [PLOT] (b) Saved -> {path_b}")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title("Confusion Matrix, Test Set")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    path_c = os.path.join(results_dir, "plot_c_confusion_matrix.png")
    plt.savefig(path_c, dpi=300)
    plt.close(fig)
    print(f"  [PLOT] (c) Saved -> {path_c}")

    test_labels_arr = np.array(test_labels)
    roc_auc_scores = {}

    fig, ax = plt.subplots(figsize=(9, 7))
    for i in range(NUM_CLASSES):
        bin_labels = (test_labels_arr == i).astype(int)
        fpr, tpr, _ = roc_curve(bin_labels, test_probs[:, i])
        roc_auc = auc(fpr, tpr)
        roc_auc_scores[CLASS_NAMES[i]] = roc_auc
        ax.plot(fpr, tpr, color=colors[i],
                label=f"{CLASS_NAMES[i]} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_title("ROC Curves, One-vs-Rest (Test Set)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc='lower right')
    ax.grid(True)
    plt.tight_layout()
    path_d = os.path.join(results_dir, "plot_d_roc_curves.png")
    plt.savefig(path_d, dpi=300)
    plt.close(fig)
    print(f"  [PLOT] (d) Saved -> {path_d}")

    fig, ax = plt.subplots(figsize=(9, 7))
    for i in range(NUM_CLASSES):
        bin_labels = (test_labels_arr == i).astype(int)
        prec_arr, rec_arr, _ = precision_recall_curve(bin_labels,
                                                      test_probs[:, i])
        ap = average_precision_score(bin_labels, test_probs[:, i])
        ax.plot(rec_arr, prec_arr, color=colors[i],
                label=f"{CLASS_NAMES[i]} (AP={ap:.3f})")
    ax.set_title("Precision-Recall Curves, One-vs-Rest (Test Set)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.tight_layout()
    path_e = os.path.join(results_dir, "plot_e_precision_recall.png")
    plt.savefig(path_e, dpi=300)
    plt.close(fig)
    print(f"  [PLOT] (e) Saved -> {path_e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    class_labels = list(roc_auc_scores.keys())
    auc_vals = list(roc_auc_scores.values())
    bars = ax.bar(class_labels, auc_vals,
                  color=[colors[i] for i in range(len(class_labels))])
    for bar, val in zip(bars, auc_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha='center', va='bottom', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class ROC-AUC Scores (Test Set)")
    ax.set_xlabel("Class")
    ax.set_ylabel("AUC")
    ax.grid(axis='y')
    plt.tight_layout()
    path_g = os.path.join(results_dir, "plot_g_auc_bar.png")
    plt.savefig(path_g, dpi=300)
    plt.close(fig)
    print(f"  [PLOT] (g) Saved -> {path_g}")

    return roc_auc_scores


# Report

def save_report(fold_results, test_acc, test_prec, test_rec, test_f1,
                test_cr, test_cm, roc_auc_scores, results_dir,
                n_train, n_test):
    """
    Write a .txt report with config summary, data split counts,
    fold-level results, mean/std across folds, test metrics,
    classification report, confusion matrix, and per-class ROC-AUC.
    """
    report_path = os.path.join(results_dir, f"{MODEL_NAME}_report.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 70)
    lines.append("  HER2-SISH IMAGE CLASSIFICATION REPORT")
    lines.append(f"  Generated : {timestamp}")
    lines.append("=" * 70)

    lines.append("\n  CONFIG SUMMARY")
    lines.append(f"  Model          : {MODEL_NAME}")
    lines.append(f"  Architecture   : MobileNetV3-Large")
    lines.append(f"  Unfrozen layers: features[-1] + classifier")
    lines.append(f"  Optimizer      : AdamW")
    lines.append(f"  Learning Rate  : {LEARNING_RATE}")
    lines.append(f"  Loss Function  : categorical cross-entropy"
                 + (", inverse-frequency class weights"
                    if USE_CLASS_WEIGHTS else ", unweighted"))
    lines.append(f"  Image Size     : {IMG_SIZE}x{IMG_SIZE}")
    lines.append(f"  Batch Size     : {BATCH_SIZE}")
    lines.append(f"  Epochs         : {EPOCHS}")
    lines.append(f"  K-Folds        : {K_FOLDS}, StratifiedGroupKFold "
                 f"grouped by source ROI")
    lines.append(f"  Augmentation   : train_aug only, offline x4 rotation "
                 f"(0/90/180/270); test = original ROIs")
    lines.append(f"  Classes        : {CLASS_NAMES}")
    lines.append(f"  Device         : {DEVICE}")
    lines.append(f"  Random seed    : {RANDOM_STATE} "
                 f"(deterministic cuDNN: {DETERMINISTIC})")
    lines.append(f"  Train dir      : {TRAIN_DIR} ({n_train} images, augmented)")
    lines.append(f"  Test dir       : {TEST_DIR} ({n_test} images, original only)")

    lines.append("\n  FOLD-LEVEL RESULTS (validation = held-out fold of TRAIN)")
    header = (f"  {'Fold':>4} | {'Best Ep':>7} | "
              f"{'Val Acc':>8} | {'Val Prec':>9} | "
              f"{'Val Rec':>8} | {'Val F1':>7}")
    lines.append(header)
    lines.append("  " + "-" * 60)
    for r in fold_results:
        lines.append(
            f"  {r['fold']:>4} | {r['best_epoch']:>7} | "
            f"{r['val_acc']:>8.4f} | {r['val_prec']:>9.4f} | "
            f"{r['val_rec']:>8.4f} | {r['val_f1']:>7.4f}"
        )

    accs  = [r['val_acc'] for r in fold_results]
    f1s   = [r['val_f1'] for r in fold_results]
    precs = [r['val_prec'] for r in fold_results]
    recs  = [r['val_rec'] for r in fold_results]
    lines.append("\n  MEAN +/- STD ACROSS FOLDS")
    lines.append(f"  Accuracy  : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    lines.append(f"  Precision : {np.mean(precs):.4f} +/- {np.std(precs):.4f}")
    lines.append(f"  Recall    : {np.mean(recs):.4f} +/- {np.std(recs):.4f}")
    lines.append(f"  F1        : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    lines.append("\n  TEST SET METRICS (held-out originals, never augmented)")
    lines.append(f"  Accuracy  : {test_acc:.4f}")
    lines.append(f"  Precision : {test_prec:.4f}")
    lines.append(f"  Recall    : {test_rec:.4f}")
    lines.append(f"  F1        : {test_f1:.4f}")
    lines.append("\n  Classification Report:")
    for line in test_cr.splitlines():
        lines.append("    " + line)

    lines.append("\n  CONFUSION MATRIX (Test Set)")
    col_header = "           " + "  ".join(f"{c[:8]:>8}" for c in CLASS_NAMES)
    lines.append(col_header)
    for i, row_cm in enumerate(test_cm):
        row_str = "  ".join(f"{v:>8}" for v in row_cm)
        lines.append(f"  {CLASS_NAMES[i][:8]:>8}  {row_str}")

    lines.append("\n  PER-CLASS ROC-AUC (Test Set)")
    for cname, auc_val in roc_auc_scores.items():
        lines.append(f"  {cname}: {auc_val:.4f}")

    lines.append("\n" + "=" * 70)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"\n  [REPORT] Saved -> {report_path}")
    return report_path


# Final summary

def print_final_summary(fold_results, test_acc, test_prec, test_rec, test_f1,
                        results_dir, checkpoint_dir, log_path):
    accs = [r['val_acc'] for r in fold_results]
    f1s  = [r['val_f1'] for r in fold_results]
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  KFold Val Accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  KFold Val F1       : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"  Test Accuracy      : {test_acc:.4f}")
    print(f"  Test Precision     : {test_prec:.4f}")
    print(f"  Test Recall        : {test_rec:.4f}")
    print(f"  Test F1            : {test_f1:.4f}")
    print(f"  Results dir        : {os.path.abspath(results_dir)}")
    print(f"  Checkpoint dir     : {os.path.abspath(checkpoint_dir)}")
    print(f"  Log file           : {os.path.abspath(log_path)}")
    print(f"{'='*60}\n")


# Main entry point

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"{MODEL_NAME}_log_{timestamp}.txt"
    sys.stdout = Logger(log_path)

    print(f"\n{'#'*60}")
    print(f"  MobileNetV3-Large HER2-SISH Classification")
    print(f"  MODEL: {MODEL_NAME}")
    print(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    set_seed(RANDOM_STATE)
    print(f"  Seed: {RANDOM_STATE}  (deterministic cuDNN: {DETERMINISTIC})")
    print(f"  Loss: categorical cross-entropy"
          f"{', inverse-frequency class weights' if USE_CLASS_WEIGHTS else ', unweighted'}")

    print_device_info()

    checkpoint_dir, results_dir = setup_folders(MODEL_NAME)

    # Load pre-split data. No internal re-split.
    print("\n[DATA] Loading pre-split train_aug and test folders...")
    X_train, y_train = collect_file_paths(TRAIN_DIR, CLASS_NAMES)
    X_test,  y_test  = collect_file_paths(TEST_DIR,  CLASS_NAMES)

    if len(X_train) == 0 or len(X_test) == 0:
        print("[ERROR] No images found. Check TRAIN_DIR / TEST_DIR and CLASS_NAMES.")
        return

    print_distribution("TRAIN (augmented)", y_train, CLASS_NAMES)
    print_distribution("TEST (original)",   y_test,  CLASS_NAMES)

    groups = build_groups(X_train)

    fold_results, all_fold_train_losses, best_hparams = run_kfold(
        X_train, y_train, groups, checkpoint_dir, results_dir)

    final_model, train_history = retrain_on_full_train(
        X_train, y_train, best_hparams, checkpoint_dir, results_dir)

    (test_loss, test_acc, test_prec, test_rec, test_f1,
     test_preds, test_labels, test_probs,
     test_cm, test_cr) = evaluate_on_test(
        final_model, X_test, y_test, results_dir, train_history)

    print("\n[PLOT] Generating visualisations...")
    roc_auc_scores = save_visualisations(
        fold_results, all_fold_train_losses, train_history,
        test_labels, test_preds, test_probs, test_cm, results_dir)

    save_report(fold_results, test_acc, test_prec, test_rec, test_f1,
                test_cr, test_cm, roc_auc_scores, results_dir,
                len(X_train), len(X_test))

    print_final_summary(fold_results, test_acc, test_prec, test_rec, test_f1,
                        results_dir, checkpoint_dir, log_path)

    if isinstance(sys.stdout, Logger):
        sys.stdout.close()
        sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()
