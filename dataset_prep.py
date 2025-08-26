# dataset_prep.py
import os
import glob
import random
import shutil
from pathlib import Path
import argparse
import math
import json

"""
Usage:
python dataset_prep.py --data_dir /path/to/TrashNet --out /path/to/workdir --val_split 0.1 --test_split 0.1
"""

CLASS_ORDER = ["plastic","paper","metal","glass","cardboard","trash"]  # adjust if different

def find_images_for_class(class_dir):
    exts = ("*.jpg","*.jpeg","*.png")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(class_dir,e)))
    files = sorted(files)
    return files

def ensure_label(img_path, class_id, labels_dir):
    # If your dataset doesn't have bounding boxes (only classification),
    # you'll need to annotate with bounding boxes first using LabelImg/Roboflow etc.
    # This function checks existence of label file and creates a dummy full-image bbox if not present.
    # WARNING: creating full-image bbox is only a placeholder; it's better to annotate real bboxes.
    p = Path(img_path)
    label_path = labels_dir / (p.stem + ".txt")
    if label_path.exists():
        return
    # Make a full-image bbox placeholder (class_id center 0.5,0.5 size 1.0)
    with open(label_path, "w") as f:
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

def main(data_dir, out, val_split=0.2, test_split=0.1):
    data_dir = Path(data_dir)
    out = Path(out)
    images_out = out / "images"
    labels_out = out / "labels"

    for split in ("train","val","test"):
        (images_out/split).mkdir(parents=True, exist_ok=True)
        (labels_out/split).mkdir(parents=True, exist_ok=True)

    all_pairs = []
    for cid, cls in enumerate(CLASS_ORDER):
        class_dir = data_dir / cls
        if not class_dir.exists():
            print(f"Warning: {class_dir} not present.")
            continue
        imgs = find_images_for_class(class_dir)
        for img in imgs:
            all_pairs.append((img, cid, cls))

    random.shuffle(all_pairs)
    n = len(all_pairs)
    n_test = math.floor(n * test_split)
    n_val = math.floor(n * val_split)

    splits = {
        "test": all_pairs[:n_test],
        "val": all_pairs[n_test:n_test+n_val],
        "train": all_pairs[n_test+n_val:]
    }

    for split, items in splits.items():
        for img_path, cid, cls in items:
            src_img = Path(img_path)
            dst_img = images_out / split / (src_img.name)
            shutil.copyfile(src_img, dst_img)
            # copy or create label
            # assume labels in YOLO txt with same basename but in same class folder? adjust if needed
            labels_dir_candidate = Path(src_img.parent)  # change if labels are elsewhere
            # Best-case: label exists with same base name and .txt in the class folder
            candidate_label = labels_dir_candidate / (src_img.stem + ".txt")
            dst_label = labels_out / split / (src_img.stem + ".txt")
            if candidate_label.exists():
                shutil.copyfile(candidate_label, dst_label)
            else:
                # fallback: create a placeholder full-image bbox (NOT ideal)
                with open(dst_label, "w") as f:
                    f.write(f"{cid} 0.5 0.5 1.0 1.0\n")

    # write data.yaml
    data_yaml = {
        "path": str(out), # root
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_ORDER
    }
    with open(out / "data.yaml", "w") as fh:
        import yaml
        fh.write(yaml.dump(data_yaml))
    print("Prepared dataset at:", out)
    print("Classes:", CLASS_ORDER)
    print("Total images:", n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    args = parser.parse_args()
    main(args.data_dir, args.out, args.val_split, args.test_split)
