# Step 1 of the data pipeline: split the original HER2-SISH ROIs into
# train and test BEFORE any augmentation is applied.
#
# This ordering is the whole point. Augmenting first and splitting afterwards
# puts rotated copies of the same ROI on both sides of the split, which leaks
# the test set into training. Run this script first, then
# augment_her2_train.py, which touches only the train partition.
#
# The split is performed over image-ID groups, never over individual files, so
# every ROI sharing an ID stays on the same side. A final assertion aborts the
# run if any ID is ever found in both partitions.
#
# Input:
#   SOURCE_DIR/
#       Amplified/
#       Non-Amplified/
#       Normal/
#
# Output:
#   OUT_ROOT/train/<class>/        original ROIs used for training
#   OUT_ROOT/test/<class>/         original ROIs held out, never augmented
#   OUT_ROOT/split_manifest.csv    every file with its ID and partition

import os
import re
import csv
import random
import shutil
import collections

SOURCE_DIR = r"C:\ML_HER2_FINAL\dataset"   # original, unaugmented ROIs
OUT_ROOT   = r"C:\ML_HER2_FINAL\split"

CLASS_NAMES = ["Amplified", "Non-Amplified", "Normal"]
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

RANDOM_STATE = 42

# Image-ID grouping.
#
# For this dataset each file is one ROI and no slide contributes more than one
# ROI, so with ID_PATTERN unset every group has size 1 and grouping is a no-op.
# The machinery is here so the no-leak guarantee holds by construction rather
# than by a one-off manual check.
#
# ID_PATTERN = None    -> the whole filename stem is the ID (one ROI per file)
# ID_PATTERN = r"..."  -> a regex whose first capture group is the ID, for
#                         filenames that encode a slide or case identifier,
#                         e.g. r"^(case\d+)_roi\d+$" turns case07_roi3 into
#                         the group "case07"
#
# Grouping is global, not per class. A single slide can legitimately yield ROIs
# of more than one class, and such a group still moves to one side as a unit.
ID_PATTERN = None

# Exact per-class test counts, so the split reproduces the published Table 2
# (161 test / 376 train out of 537 source ROIs). Set TEST_COUNTS to None to
# fall back to a proportional split using TEST_FRACTION instead.
TEST_COUNTS = {
    "Amplified":     34,
    "Non-Amplified": 37,
    "Normal":        90,
}
TEST_FRACTION = 0.30


def extract_image_id(filename):
    """
    Return the grouping ID for one file. With ID_PATTERN unset the stem is the
    ID, which makes each file its own group. A file that does not match a set
    ID_PATTERN raises, rather than silently falling back to a per-file group
    and defeating the point of grouping.
    """
    stem = os.path.splitext(filename)[0]
    if ID_PATTERN is None:
        return stem
    match = re.match(ID_PATTERN, stem)
    if match is None:
        raise ValueError(
            f"Filename '{filename}' does not match ID_PATTERN "
            f"{ID_PATTERN!r}. Fix the pattern or the filename; falling back "
            f"to a per-file ID here would hide real ID collisions."
        )
    return match.group(1)


def list_class_files(source_dir, class_name):
    """Return sorted file names for one class folder."""
    folder = os.path.join(source_dir, class_name)
    if not os.path.isdir(folder):
        print(f"[WARN] Class folder not found: {folder}")
        return []
    return sorted(
        fname for fname in os.listdir(folder)
        if fname.lower().endswith(VALID_EXTENSIONS)
    )


def make_dirs(out_root, class_names):
    """Create train/<class> and test/<class> under out_root."""
    for part in ("train", "test"):
        for cname in class_names:
            os.makedirs(os.path.join(out_root, part, cname), exist_ok=True)


def build_groups(source_dir, class_names):
    """
    Walk every class folder and group all files by image ID, across classes.
    Returns an ordered mapping: image_id -> list of (class_name, filename).
    """
    groups = collections.OrderedDict()
    for class_name in class_names:
        for fname in list_class_files(source_dir, class_name):
            gid = extract_image_id(fname)
            groups.setdefault(gid, []).append((class_name, fname))
    return groups


def assign_stratum(members):
    """
    The class a group is counted against when filling per-class test budgets.
    For a group whose files all share one class this is simply that class. For
    a group spanning classes it is the most common one, with ties broken by
    CLASS_NAMES order so the choice is deterministic.
    """
    counts = collections.Counter(cname for cname, _ in members)
    top = max(counts.values())
    for cname in CLASS_NAMES:
        if counts.get(cname) == top:
            return cname
    return members[0][0]


def split_groups(groups, rng):
    """
    Split ID groups into train and test. Whole groups move together, so no ID
    can land on both sides. Per-class test budgets are filled in files, and a
    group is only taken for test while it fits entirely in the budget left for
    its stratum.
    Returns (train_ids, test_ids) as lists of image IDs.
    """
    # Bucket group IDs by stratum, and record how many files each one carries
    by_stratum = collections.defaultdict(list)
    file_counts = {}
    for gid, members in groups.items():
        by_stratum[assign_stratum(members)].append(gid)
        file_counts[gid] = len(members)

    train_ids, test_ids = [], []

    for class_name in CLASS_NAMES:
        gids = by_stratum.get(class_name, [])
        n_files = sum(file_counts[g] for g in gids)
        if n_files == 0:
            continue

        if TEST_COUNTS is not None:
            budget = TEST_COUNTS.get(class_name)
            if budget is None:
                raise KeyError(
                    f"TEST_COUNTS has no entry for class '{class_name}'")
            if budget > n_files:
                raise ValueError(
                    f"TEST_COUNTS[{class_name}]={budget} exceeds the "
                    f"{n_files} files available for that class"
                )
        else:
            budget = int(round(n_files * TEST_FRACTION))

        shuffled = list(gids)
        rng.shuffle(shuffled)

        taken = 0
        for gid in shuffled:
            size = file_counts[gid]
            if taken + size <= budget:
                test_ids.append(gid)
                taken += size
            else:
                train_ids.append(gid)

        if taken != budget:
            print(f"  [NOTE] {class_name}: test target {budget} files, "
                  f"reached {taken} without splitting any ID group.")

    return train_ids, test_ids


def assert_no_id_overlap(manifest_rows):
    """
    Final guard: abort if any ID appears in both partitions. With the grouped
    split above this cannot trigger, which is the point. It exists so the
    script fails loudly rather than leaking if the split logic, the filenames
    or ID_PATTERN are ever changed.
    """
    by_partition = collections.defaultdict(set)
    for row in manifest_rows:
        by_partition[row['partition']].add(row['image_id'])

    overlap = by_partition['train'] & by_partition['test']
    if overlap:
        sample = sorted(overlap)[:10]
        raise AssertionError(
            f"ID LEAK: {len(overlap)} image ID(s) appear in both train and "
            f"test. First few: {sample}. The split was NOT written correctly; "
            f"do not train on this output."
        )
    print(f"  [CHECK] No ID appears in both partitions "
          f"({len(by_partition['train'])} train IDs, "
          f"{len(by_partition['test'])} test IDs).")


def main():
    if not os.path.isdir(SOURCE_DIR):
        print(f"[ERROR] SOURCE_DIR does not exist: {SOURCE_DIR}")
        return

    rng = random.Random(RANDOM_STATE)
    make_dirs(OUT_ROOT, CLASS_NAMES)

    print(f"[SPLIT] Source : {SOURCE_DIR}")
    print(f"[SPLIT] Output : {OUT_ROOT}")
    print(f"[SPLIT] Seed   : {RANDOM_STATE}")
    print(f"[SPLIT] ID rule: "
          f"{'filename stem (one ROI per file)' if ID_PATTERN is None else ID_PATTERN!r}")
    print(f"[SPLIT] Mode   : "
          f"{'exact per-class counts' if TEST_COUNTS else f'{TEST_FRACTION:.0%} proportional'}")
    print()

    groups = build_groups(SOURCE_DIR, CLASS_NAMES)
    if not groups:
        print("[ERROR] No images found. Check SOURCE_DIR and CLASS_NAMES.")
        return

    n_files = sum(len(m) for m in groups.values())
    multi = sum(1 for m in groups.values() if len(m) > 1)
    spanning = sum(1 for m in groups.values()
                   if len({c for c, _ in m}) > 1)
    print(f"[SPLIT] {n_files} files in {len(groups)} ID groups "
          f"({multi} with more than one file, {spanning} spanning classes)")
    print()

    train_ids, test_ids = split_groups(groups, rng)

    manifest_rows = []
    totals = collections.Counter()
    per_class = collections.defaultdict(collections.Counter)

    for part, ids in (("train", train_ids), ("test", test_ids)):
        for gid in ids:
            for class_name, fname in groups[gid]:
                src = os.path.join(SOURCE_DIR, class_name, fname)
                dst = os.path.join(OUT_ROOT, part, class_name, fname)
                shutil.copy2(src, dst)
                manifest_rows.append({
                    'partition': part,
                    'class': class_name,
                    'image_id': gid,
                    'filename': fname,
                    'source_path': src,
                    'dest_path': dst,
                })
                totals[part] += 1
                per_class[class_name][part] += 1

    for class_name in CLASS_NAMES:
        c = per_class[class_name]
        total = c['train'] + c['test']
        if total:
            print(f"  {class_name:<16} total={total:>4}  "
                  f"train={c['train']:>4}  test={c['test']:>4}")

    manifest_path = os.path.join(OUT_ROOT, "split_manifest.csv")
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['partition', 'class', 'image_id', 'filename',
                           'source_path', 'dest_path'])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    assert_no_id_overlap(manifest_rows)
    print()
    print(f"[SPLIT] TRAIN total : {totals['train']}")
    print(f"[SPLIT] TEST  total : {totals['test']}")
    print(f"[SPLIT] GRAND total : {totals['train'] + totals['test']}")
    print(f"[SPLIT] Manifest    : {manifest_path}")
    print()
    print("Next step: run augment_her2_train.py to build split/train_aug.")
    print("The test partition stays untouched and unaugmented.")


if __name__ == '__main__':
    main()
