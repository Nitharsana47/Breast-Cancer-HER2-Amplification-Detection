# Step 2 of the data pipeline: offline augmentation of the TRAIN partition only.
#
# Run this after split_her2.py. It reads split/train and writes split/train_aug
# with a 4x expansion per image: the original plus 90, 180 and 270 degree
# rotations. The test partition is never read or written here, so no rotated
# copy of a test ROI can ever reach the training set.
#
# 376 train originals -> 1,504 augmented training images.
#
# Input:
#   IN_DIR/<class>/     original training ROIs from split_her2.py
#
# Output:
#   OUT_DIR/<class>/    original + rot90 + rot180 + rot270
#   OUT_DIR/augmentation_manifest.csv   every output file and its source

import os
import csv
import collections

import cv2

IN_DIR  = r"C:\ML_HER2_FINAL\split\train"
OUT_DIR = r"C:\ML_HER2_FINAL\split\train_aug"

CLASS_NAMES = ["Amplified", "Non-Amplified", "Normal"]
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

# suffix -> cv2 rotation code. None means write the original unchanged.
ROTATIONS = [
    ("orig",   None),
    ("rot90",  cv2.ROTATE_90_CLOCKWISE),
    ("rot180", cv2.ROTATE_180),
    ("rot270", cv2.ROTATE_90_COUNTERCLOCKWISE),
]


def augment_one_image(src_path, dst_folder, stem, ext):
    """
    Write the original plus three rotations of one image.
    Returns a list of (output_filename, variant) pairs actually written.
    """
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[WARN] Could not read {src_path}, skipping.")
        return []

    written = []
    for suffix, rot_code in ROTATIONS:
        out_img = img if rot_code is None else cv2.rotate(img, rot_code)
        out_name = f"{stem}_{suffix}{ext}"
        out_path = os.path.join(dst_folder, out_name)
        ok = cv2.imwrite(out_path, out_img)
        if not ok:
            print(f"[WARN] Could not write {out_path}, skipping.")
            continue
        written.append((out_name, suffix))
    return written


def main():
    if not os.path.isdir(IN_DIR):
        print(f"[ERROR] IN_DIR does not exist: {IN_DIR}")
        print("Run split_her2.py first.")
        return

    print(f"[AUG] Input  : {IN_DIR}")
    print(f"[AUG] Output : {OUT_DIR}")
    print(f"[AUG] Variants per image: "
          f"{', '.join(s for s, _ in ROTATIONS)}  (x{len(ROTATIONS)})")
    print()

    manifest_rows = []
    in_totals = collections.Counter()
    out_totals = collections.Counter()

    for class_name in CLASS_NAMES:
        src_folder = os.path.join(IN_DIR, class_name)
        dst_folder = os.path.join(OUT_DIR, class_name)

        if not os.path.isdir(src_folder):
            print(f"[WARN] Class folder not found: {src_folder}")
            continue
        os.makedirs(dst_folder, exist_ok=True)

        files = sorted(
            fname for fname in os.listdir(src_folder)
            if fname.lower().endswith(VALID_EXTENSIONS)
        )

        for fname in files:
            stem, ext = os.path.splitext(fname)
            src_path = os.path.join(src_folder, fname)
            written = augment_one_image(src_path, dst_folder, stem, ext)

            for out_name, variant in written:
                manifest_rows.append({
                    'class': class_name,
                    'source_file': fname,
                    'output_file': out_name,
                    'variant': variant,
                })

            in_totals[class_name] += 1
            out_totals[class_name] += len(written)

        print(f"  {class_name:<16} in={in_totals[class_name]:>4}  "
              f"out={out_totals[class_name]:>5}")

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "augmentation_manifest.csv")
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['class', 'source_file', 'output_file', 'variant'])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print(f"[AUG] Source images  : {sum(in_totals.values())}")
    print(f"[AUG] Augmented out  : {sum(out_totals.values())}")
    print(f"[AUG] Manifest       : {manifest_path}")
    print()
    print("The test partition was not read or modified by this script.")
    print("Next step: run any script in models/ against train_aug and test.")


if __name__ == '__main__':
    main()
