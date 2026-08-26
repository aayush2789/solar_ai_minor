import os
import glob
import argparse
from collections import defaultdict

CLASSES = ['dust', 'defect', 'bird drop', 'snow', 'clean', 'leaf', 'bird feather']

def count_classes(dataset_dir):
    """
    Parses a YOLO format dataset directory and returns detailed statistics.
    Returns:
        dict: {split: {'images': count, 'labels': count, 'classes': {class_id: count}}}
    """
    splits = ['train', 'valid', 'test']
    stats = {}

    for split in splits:
        img_dir = os.path.join(dataset_dir, split, "images")
        lbl_dir = os.path.join(dataset_dir, split, "labels")

        img_count = len(glob.glob(os.path.join(img_dir, "*.*"))) if os.path.exists(img_dir) else 0
        lbl_files = glob.glob(os.path.join(lbl_dir, "*.txt")) if os.path.exists(lbl_dir) else []
        lbl_count = len(lbl_files)

        cls_counts = defaultdict(int)
        valid_boxes = 0
        corrupt_lines = 0

        for lbl_path in lbl_files:
            try:
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(parts[0])
                            # Check normalized bbox coordinates
                            x, y, w, h = map(float, parts[1:5])
                            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0:
                                cls_counts[cls_id] += 1
                                valid_boxes += 1
                            else:
                                corrupt_lines += 1
                        except ValueError:
                            corrupt_lines += 1
            except Exception as e:
                print(f"Error reading {lbl_path}: {e}")

        stats[split] = {
            'images': img_count,
            'labels': lbl_count,
            'classes': cls_counts,
            'valid_boxes': valid_boxes,
            'corrupt_lines': corrupt_lines
        }

    return stats

def print_dataset_report(dataset_dir):
    """Prints a formatted report of dataset statistics."""
    stats = count_classes(dataset_dir)
    print(f"\n================ DATASET REPORT: {dataset_dir} ================")
    
    total_imgs = 0
    total_lbls = 0
    total_cls = defaultdict(int)

    for split in ['train', 'valid', 'test']:
        if split not in stats:
            continue
        s = stats[split]
        total_imgs += s['images']
        total_lbls += s['labels']
        print(f"\nSplit: {split}")
        print(f"  Images: {s['images']}")
        print(f"  Labels (files): {s['labels']}")
        print(f"  Class breakdown:")
        for cid in range(len(CLASSES)):
            cname = CLASSES[cid]
            cnt = s['classes'].get(cid, 0)
            total_cls[cid] += cnt
            print(f"    - {cname} (ID {cid}): {cnt}")

    print("\n--- Totals ---")
    print(f"Total Images: {total_imgs}")
    print(f"Total Labels (files): {total_lbls}")
    print(f"Total Class breakdown:")
    for cid in range(len(CLASSES)):
        cname = CLASSES[cid]
        print(f"  - {cname} (ID {cid}): {total_cls[cid]}")
    print("=================================================================\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze YOLO dataset statistics")
    parser.add_argument("--data-dir", type=str, default="/home/cse/solar_ai/dataset", help="Dataset root directory")
    args = parser.parse_args()

    if os.path.exists(args.data-dir):
        print_dataset_report(args.data_dir)
    else:
        print(f"Directory not found: {args.data_dir}")
