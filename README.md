# HER2-SISH Histopathological Image Classification

Three-class classification of HER2 silver in-situ hybridisation (SISH) regions
of interest: **Amplified**, **Non-Amplified**, **Normal**.

Six architectures are compared under one identical training and evaluation
protocol.

## Protocol

The split happens **before** augmentation. This is the corrected protocol; an
earlier version augmented first and split afterwards, which placed rotated
copies of the same ROI on both sides of the split and leaked the test set into
training. Every number produced by this repository uses split-then-augment.

```
537 original ROIs
     │
     ├── split_her2.py  (stratified, seed 42, originals only)
     │
     ├── split/train   376 originals
     │        │
     │        └── augment_her2_train.py  (×4: 0°, 90°, 180°, 270°)
     │                 │
     │                 └── split/train_aug   1,504 images
     │
     └── split/test    161 originals, never augmented, never split again
```

Per-class counts:

| Class | Source | Train | Test | Train augmented |
|---|---|---|---|---|
| Amplified | 113 | 79 | 34 | 316 |
| Non-Amplified | 124 | 87 | 37 | 348 |
| Normal | 300 | 210 | 90 | 840 |
| **Total** | **537** | **376** | **161** | **1,504** |

Each training script then runs three phases:

1. **K-fold CV** — StratifiedGroupKFold(5, shuffle, seed 42) over `train_aug`
   only, grouped by source ROI so all four rotations of an image stay inside
   the same fold. Validation is a held-out fold of the training pool. The test
   set is not touched.
2. **Phase 2** — retrain a fresh model on all of `train_aug`.
3. **Phase 3** — evaluate once on `split/test`, original unaugmented ROIs.

## Layout

```
data_prep/
  split_her2.py            537 originals -> split/train + split/test, grouped by image ID
  augment_her2_train.py    split/train -> split/train_aug (×4 rotation)

models/
  mobilenet_v3_large_lastblock.py      MobileNetV3-Large
  densenet121_denseblock4.py           DenseNet121
  deit_small_patch16_224_block11.py    DeiT-Small
  vit_small_patch16_224_block11.py     ViT-Small
  densenet121_deit_small_hybrid.py     DenseNet121 + DeiT-Small, early fusion
  densenet121_vit_small_hybrid.py      DenseNet121 + ViT-Small, early fusion
```

## Fine-tuning strategy per model

| Model | Trainable |
|---|---|
| MobileNetV3-Large | `features[-1]` + classifier |
| DenseNet121 | `denseblock4` + `norm5` + classifier |
| DeiT-Small | `blocks[11]` + head |
| ViT-Small | `blocks[11]` + head |
| DN121 + DeiT-Small | `denseblock4` + `norm5`, `blocks[10:12]` + `norm`, `proj`, CLS, pos_embed, classifier |
| DN121 + ViT-Small | same as above |

The two hybrids share a frozen DenseNet121 `features[0:10]` trunk. Both
branches read the same feature map (early fusion), reduce to 1024-d and 384-d
vectors respectively, and are concatenated to 1408-d before a single
`Linear(1408, 3)` head.

## Shared hyperparameters

| | |
|---|---|
| Image size | 224 × 224 |
| Batch size | 16 |
| Epochs | 30 |
| Optimizer | AdamW |
| Learning rate | 3e-5 |
| Loss | categorical cross-entropy (`USE_CLASS_WEIGHTS = False`) |
| K-folds | 5, StratifiedGroupKFold grouped by source ROI |
| Early stopping | patience 5 on validation F1 (K-fold phase) |
| Metric averaging | macro |
| Seed | 42 (Python, NumPy, torch CPU + CUDA; cuDNN deterministic) |

## Running

```bash
pip install -r requirements.txt

python data_prep/split_her2.py
python data_prep/augment_her2_train.py

python models/densenet121_denseblock4.py
python models/mobilenet_v3_large_lastblock.py
python models/deit_small_patch16_224_block11.py
python models/vit_small_patch16_224_block11.py
python models/densenet121_deit_small_hybrid.py
python models/densenet121_vit_small_hybrid.py
```

Edit `SOURCE_DIR` / `OUT_ROOT` in `split_her2.py` and `TRAIN_DIR` / `TEST_DIR`
at the top of each model script to match your machine. The defaults assume
`C:\ML_HER2_FINAL\`.

## Outputs

Each model writes to its own directories, keyed on `MODEL_NAME`:

```
checkpoints/<model>/
  <model>_fold_N.pth         latest per fold, used to resume mid-fold
  <model>_fold_N_best.pth    best validation F1 per fold
  <model>_fold_0.pth         Phase 2 resume checkpoint

results/<model>/
  epoch_results.csv          per-epoch train and val metrics, every fold
  fold_results.csv           best-epoch summary per fold
  combined_training.csv      Phase 2 per-epoch train metrics
  test_results.csv           test metrics + last-epoch train metrics
  <model>_report.txt         full text report
  <model>_final_model.pth    Phase 2 final weights
  plot_a_fold_loss_curves.png
  plot_b_train_test_accuracy.png
  plot_c_confusion_matrix.png
  plot_d_roc_curves.png
  plot_e_precision_recall.png
  plot_g_auc_bar.png
```

All plots are 300 DPI. All six models write identical CSV columns and plot
filenames, so results can be collated across models without special-casing.

A timestamped `<model>_log_<YYYYmmdd_HHMMSS>.txt` is written to the working
directory, capturing everything printed to the console.

## Guarantees enforced in code

Three things fail loudly rather than degrading quietly:

- `split_her2.py` aborts if any image ID lands in both train and test.
- `build_groups()` warns if no filename carries a rotation suffix, which would
  mean grouped CV has silently become plain CV.
- Each fold asserts that its train and validation groups are disjoint before
  training starts.

## Reproducibility

`set_seed()` seeds Python, NumPy and torch on CPU and GPU, and pins cuDNN to
deterministic kernels. It is called at the start of each run, again per fold
with `RANDOM_STATE + fold_idx`, and again before Phase 2, so a fold is
reproducible whether the run reached it directly or resumed into it. Set
`DETERMINISTIC = False` at the top of a script to let cuDNN autotune, which is
faster but no longer bit-reproducible.

Runs made before seeding was added cannot be recreated exactly.

`split_her2.py` splits over image-ID groups rather than individual files, and
asserts at the end that no ID appears in both partitions, aborting if one does.
With `ID_PATTERN = None` every file is its own ID, which matches this dataset
(one ROI per file, no slide contributing more than one ROI), so grouping is a
no-op and the guard simply confirms it. If a future dataset version encodes a
slide or case identifier in the filename, set `ID_PATTERN` to a regex capturing
it and whole slides will move to one side as a unit, including slides that
contribute ROIs of more than one class.

## Resuming

Every script resumes if interrupted. A fold counts as complete only once its
summary row is in `fold_results.csv`; a fold stopped mid-training resumes from
its last epoch checkpoint rather than being skipped. Phase 2 resumes from
`<model>_fold_0.pth`. CSV headers are written only when a file does not
already exist, so a restart appends instead of overwriting.

## Known limitations

Two things worth stating plainly rather than leaving for a reviewer to find.

**Phase 2 "early stopping" monitors training F1.** There is no held-out set in
Phase 2, since the model trains on all of `train_aug`. Training F1 almost
always keeps improving, so the patience counter effectively never fires and
the phase runs the full 30 epochs. Describe this as a fixed-epoch retrain, not
as early stopping. All six models behave identically here, so cross-model
comparison is unaffected.

**Class weighting is a switch, currently off.** `USE_CLASS_WEIGHTS = False`
at the top of each script gives plain categorical cross-entropy, every sample
counted equally. Setting it to `True` applies inverse-frequency weights
(`w_c = N / (K * n_c)`), recomputed per fold from that fold's training labels
and again for Phase 2, which compensates for Normal being roughly 2.4x the
size of either amplification class. Both settings are categorical
cross-entropy; the flag only changes the per-class scaling. Whichever you run,
make Table 4 of the manuscript say the same thing.

**Grouped folds lower the validation column, correctly.** Folds are grouped by
source ROI, so validation scores sit below what an ungrouped split would
report. Ungrouped, 1,087 group overlaps occur across the five folds, each one a
rotation of a validation ROI sitting in the training set. Test metrics are
unaffected either way, since the test set is separated before augmentation.

