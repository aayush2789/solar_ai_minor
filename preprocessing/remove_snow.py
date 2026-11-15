#!/usr/bin/env python3
"""
Script to remove snow images and remap remaining classes for YOLO training.
Original Classes:
  0: dust
  1: defect
  2: bird drop
  3: snow  --> REMOVED
  4: clean -> 3
  5: leaf -> 4
  6: bird feather -> 5

New Classes (nc: 6):
  0: dust
  1: defect
  2: bird drop
  3: clean
  4: leaf
  5: bird feather
"""

import os
import shutil
import argparse
from pathlib import Path
from collections import Counter
import yaml

OLD_CLASSES = ['dust', 'defect', 'bird drop', 'snow', 'clean', 'leaf', 'bird feather']
NEW_CLASSES = ['dust', 'defect', 'bird drop', 'clean', 'leaf', 'bird feather']

CLASS_REMAP = {
    0: 0,
    1: 1,
    2: 2,
    # 3 is removed
    4: 3,
    5: 4,
    6: 5
}

def remove_snow_and_remap(
    dataset_dir: str,
    backup_dir: str,
    dry_run: bool = False
):
    dataset_path = Path(dataset_dir).resolve()
    backup_path = Path(backup_dir).resolve()

    print("=" * 80)
    print(f"REMOVE SNOW AND REMAP CLASSES {'[DRY RUN]' if dry_run else ''}")
    print("=" * 80)
    print(f"Dataset Path : {dataset_path}")
    print(f"Backup Path  : {backup_path}")
    print(f"Old Classes  : {OLD_CLASSES}")
    print(f"New Classes  : {NEW_CLASSES}")
    print("-" * 80)

    splits = ['train', 'valid', 'test']
    total_snow_images = 0
    total_remapped_files = 0
    total_remaining_files = 0

    for split in splits:
        img_dir = dataset_path / split / 'images'
        lbl_dir = dataset_path / split / 'labels'

        if not lbl_dir.exists():
            print(f"Skipping {split}: {lbl_dir} does not exist.")
            continue

        backup_img_dir = backup_path / split / 'images'
        backup_lbl_dir = backup_path / split / 'labels'
        if not dry_run:
            backup_img_dir.mkdir(parents=True, exist_ok=True)
            backup_lbl_dir.mkdir(parents=True, exist_ok=True)

        snow_files = []
        keep_files = []

        old_class_counts = Counter()
        new_class_counts = Counter()

        for entry in os.scandir(lbl_dir):
            if not entry.name.endswith('.txt'):
                continue

            with open(entry.path, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]

            has_snow = False
            for line in lines:
                parts = line.split()
                if parts:
                    try:
                        cid = int(parts[0])
                        old_class_counts[cid] += 1
                        if cid == 3:
                            has_snow = True
                    except ValueError:
                        continue

            stem = Path(entry.name).stem
            img_file = img_dir / f"{stem}.jpg"
            if not img_file.exists():
                for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                    alt = img_dir / f"{stem}{ext}"
                    if alt.exists():
                        img_file = alt
                        break

            if has_snow:
                snow_files.append((entry.path, img_file, lines))
            else:
                keep_files.append((entry.path, lines))

        print(f"\nSplit: {split}")
        print(f"  Total label files : {len(snow_files) + len(keep_files)}")
        print(f"  Snow files to move: {len(snow_files)}")
        print(f"  Retained files    : {len(keep_files)}")

        # Process snow files
        for lbl_path, img_file, _ in snow_files:
            total_snow_images += 1
            if not dry_run:
                shutil.move(str(lbl_path), str(backup_lbl_dir / Path(lbl_path).name))
                if img_file.exists():
                    shutil.move(str(img_file), str(backup_img_dir / img_file.name))

        # Process retained files and remap
        split_remapped = 0
        for lbl_path, lines in keep_files:
            total_remaining_files += 1
            needs_update = False
            new_lines = []
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                try:
                    old_cid = int(parts[0])
                    if old_cid in CLASS_REMAP:
                        new_cid = CLASS_REMAP[old_cid]
                        if new_cid != old_cid:
                            needs_update = True
                        new_parts = [str(new_cid)] + parts[1:]
                        new_lines.append(" ".join(new_parts))
                        new_class_counts[new_cid] += 1
                    else:
                        print(f"Warning: Unknown class ID {old_cid} in {lbl_path}")
                except ValueError:
                    continue

            if needs_update:
                split_remapped += 1
                total_remapped_files += 1
                if not dry_run:
                    with open(lbl_path, 'w', encoding='utf-8') as f:
                        for nl in new_lines:
                            f.write(nl + "\n")

        print(f"  Files remapped    : {split_remapped}")
        print(f"  Old class counts  : {dict(sorted(old_class_counts.items()))}")
        print(f"  New class counts  : {dict(sorted(new_class_counts.items()))}")

        # Delete any Ultralytics cache files in the split
        if not dry_run:
            for cache_file in [lbl_dir / 'labels.cache', dataset_path / split / 'labels.cache']:
                if cache_file.exists():
                    cache_file.unlink()
                    print(f"  Removed stale cache: {cache_file}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total snow images removed : {total_snow_images}")
    print(f"Total retained images     : {total_remaining_files}")
    print(f"Total files remapped      : {total_remapped_files}")

    # Update data.yaml
    if not dry_run:
        data_yaml_paths = [
            dataset_path / 'data.yaml',
            Path('/home/cse/solar_ai/configs/data.yaml')
        ]
        for y_path in data_yaml_paths:
            if y_path.exists():
                with open(y_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f) or {}
                cfg['nc'] = len(NEW_CLASSES)
                cfg['names'] = {i: name for i, name in enumerate(NEW_CLASSES)}
                with open(y_path, 'w', encoding='utf-8') as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                print(f"Updated dataset YAML: {y_path}")

    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Remove snow images and remap classes")
    parser.add_argument("--dataset", type=str, default="/home/cse/solar_ai/merged_dataset_v3", help="Dataset directory")
    parser.add_argument("--backup", type=str, default="/home/cse/solar_ai/solar_datasets/removed_snow_images", help="Backup directory for snow images")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without modifying files")
    args = parser.parse_args()

    remove_snow_and_remap(args.dataset, args.backup, dry_run=args.dry_run)

if __name__ == '__main__':
    main()
