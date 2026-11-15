#!/usr/bin/env python3
"""
Targeted Offline Dataset Rebalancing & Minority Class Oversampling
for Solar Panel Defect Detection (YOLO format)

Generates photometric, geometric, and affine augmentations specifically
targeted at underrepresented classes (e.g. bird feather, bird drop, defect, clean)
to equalize training distributions while preserving validation and test splits.
"""

import os
import glob
import shutil
import random
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import yaml
from tqdm import tqdm

CLASSES = ['dust', 'defect', 'bird drop', 'clean', 'leaf', 'bird feather']

def parse_yolo_label(lbl_path: str) -> List[Dict]:
    """Parses YOLO label text file into [{'class': int, 'bbox': [x, y, w, h]}]."""
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                cls_id = int(parts[0])
                bbox = list(map(float, parts[1:5]))
                boxes.append({'class': cls_id, 'bbox': bbox})
            except ValueError:
                continue
    return boxes

def save_yolo_label(lbl_path: str, boxes: List[Dict]):
    """Saves bounding boxes to a YOLO label file."""
    with open(lbl_path, 'w', encoding='utf-8') as f:
        for b in boxes:
            f.write(f"{b['class']} {b['bbox'][0]:.6f} {b['bbox'][1]:.6f} {b['bbox'][2]:.6f} {b['bbox'][3]:.6f}\n")

def augment_image_and_boxes(img: np.ndarray, boxes: List[Dict], aug_type: str = 'flip_h') -> Tuple[np.ndarray, List[Dict]]:
    """
    Applies image augmentations and recalculates normalized bounding box coordinates.
    aug_type options: 'flip_h', 'flip_v', 'color_jitter', 'affine', 'brightness'
    """
    h, w, _ = img.shape
    aug_img = img.copy()
    aug_boxes = []

    if aug_type == 'flip_h':
        aug_img = cv2.flip(img, 1)
        for b in boxes:
            cls_id = b['class']
            x, y, bw, bh = b['bbox']
            new_x = round(1.0 - x, 6)
            aug_boxes.append({'class': cls_id, 'bbox': [new_x, y, bw, bh]})

    elif aug_type == 'flip_v':
        aug_img = cv2.flip(img, 0)
        for b in boxes:
            cls_id = b['class']
            x, y, bw, bh = b['bbox']
            new_y = round(1.0 - y, 6)
            aug_boxes.append({'class': cls_id, 'bbox': [x, new_y, bw, bh]})

    elif aug_type == 'color_jitter':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_shift = random.uniform(-12, 12)
        s_scale = random.uniform(0.75, 1.25)
        v_scale = random.uniform(0.75, 1.25)

        hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)

        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        for b in boxes:
            aug_boxes.append({'class': b['class'], 'bbox': list(b['bbox'])})

    elif aug_type == 'brightness':
        factor = random.uniform(0.7, 1.3)
        aug_img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        for b in boxes:
            aug_boxes.append({'class': b['class'], 'bbox': list(b['bbox'])})

    elif aug_type == 'affine':
        dx = random.uniform(-0.04, 0.04)
        dy = random.uniform(-0.04, 0.04)
        scale = random.uniform(0.92, 1.08)

        M = np.float32([[scale, 0, dx * w], [0, scale, dy * h]])
        aug_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        for b in boxes:
            cls_id = b['class']
            x, y, bw, bh = b['bbox']
            new_x = float(np.clip((x * scale) + dx, 0.0, 1.0))
            new_y = float(np.clip((y * scale) + dy, 0.0, 1.0))
            new_bw = float(np.clip(bw * scale, 0.001, 1.0))
            new_bh = float(np.clip(bh * scale, 0.001, 1.0))
            aug_boxes.append({'class': cls_id, 'bbox': [new_x, new_y, new_bw, new_bh]})

    else:
        aug_boxes = [dict(b) for b in boxes]

    return aug_img, aug_boxes

def rebalance_dataset(
    source_dir: str,
    output_dir: str,
    target_multipliers: Optional[Dict[int, int]] = None
) -> str:
    """
    Creates a new balanced dataset directory where minority training instances
    are augmented with offline targeted synthesis.
    Validation and test splits are preserved untouched.
    """
    source_path = Path(source_dir).resolve()
    output_path = Path(output_dir).resolve()

    if target_multipliers is None:
        # Default multipliers for minority classes
        # class 5 (bird feather): ~8-10x
        # class 2 (bird drop): ~3x
        # class 1 (defect): ~2x
        # class 3 (clean): ~2x
        target_multipliers = {
            5: 8,  # bird feather (102 -> ~900)
            2: 3,  # bird drop (2214 -> ~6600)
            1: 2,  # defect (2200 -> ~4400)
            3: 2   # clean (2277 -> ~4500)
        }

    print("\n" + "=" * 80)
    print("      OFFLINE TARGETED REBALANCING & MINORITY OVERSAMPLING")
    print("=" * 80)
    print(f"Source Dataset: {source_path}")
    print(f"Output Dataset: {output_path}")
    print(f"Target Multipliers: {target_multipliers}")

    # Prepare directories
    for split in ['train', 'valid', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 1. Copy valid and test splits untouched
    for split in ['valid', 'test']:
        src_img_dir = source_path / split / 'images'
        src_lbl_dir = source_path / split / 'labels'
        dst_img_dir = output_path / split / 'images'
        dst_lbl_dir = output_path / split / 'labels'

        if src_img_dir.exists():
            for f in src_img_dir.iterdir():
                if f.is_file() and not (dst_img_dir / f.name).exists():
                    shutil.copy2(f, dst_img_dir / f.name)
        if src_lbl_dir.exists():
            for f in src_lbl_dir.iterdir():
                if f.is_file() and not (dst_lbl_dir / f.name).exists():
                    shutil.copy2(f, dst_lbl_dir / f.name)

    # 2. Process train split
    src_train_img = source_path / 'train' / 'images'
    src_train_lbl = source_path / 'train' / 'labels'
    dst_train_img = output_path / 'train' / 'images'
    dst_train_lbl = output_path / 'train' / 'labels'

    train_images = sorted([f for f in src_train_img.iterdir() if f.is_file()])
    print(f"\nProcessing {len(train_images)} training images...")

    aug_methods = ['flip_h', 'color_jitter', 'affine', 'brightness', 'flip_v']
    generated_count = 0
    copied_count = 0

    initial_class_counts = Counter()
    final_class_counts = Counter()

    for img_file in tqdm(train_images, desc="Rebalancing train split"):
        stem = img_file.stem
        lbl_file = src_train_lbl / f"{stem}.txt"

        # Always copy original
        shutil.copy2(img_file, dst_train_img / img_file.name)
        if lbl_file.exists():
            shutil.copy2(lbl_file, dst_train_lbl / lbl_file.name)
        copied_count += 1

        boxes = parse_yolo_label(str(lbl_file)) if lbl_file.exists() else []
        for b in boxes:
            initial_class_counts[b['class']] += 1
            final_class_counts[b['class']] += 1

        if not boxes:
            continue

        classes_in_img = {b['class'] for b in boxes}
        max_augs = 0
        for cid in classes_in_img:
            if cid in target_multipliers:
                max_augs = max(max_augs, target_multipliers[cid])

        if max_augs == 0:
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            continue

        for i in range(max_augs):
            aug_type = aug_methods[i % len(aug_methods)]
            aug_img, aug_boxes = augment_image_and_boxes(img, boxes, aug_type=aug_type)

            new_stem = f"{stem}_aug_{aug_type}_{i+1}"
            new_img_path = dst_train_img / f"{new_stem}{img_file.suffix}"
            new_lbl_path = dst_train_lbl / f"{new_stem}.txt"

            cv2.imwrite(str(new_img_path), aug_img)
            save_yolo_label(str(new_lbl_path), aug_boxes)
            generated_count += 1

            for b in aug_boxes:
                final_class_counts[b['class']] += 1

    # Write data.yaml in output directory
    data_yaml_content = {
        'path': str(output_path),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(CLASSES),
        'names': {idx: name for idx, name in enumerate(CLASSES)}
    }
    out_yaml = output_path / 'data.yaml'
    with open(out_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False, sort_keys=False)

    print("\n" + "=" * 80)
    print("                    REBALANCING SUMMARY")
    print("=" * 80)
    print(f"Original Training Images Copied  : {copied_count}")
    print(f"Targeted Synthetic Images Created: {generated_count}")
    print(f"Total Final Training Images      : {copied_count + generated_count}")
    print("\nClass Box Count Comparison:")
    print(f"{'Class':<15} {'Before':<12} {'After':<12} {'Gain':<10}")
    print("-" * 55)
    for cid, cname in enumerate(CLASSES):
        before = initial_class_counts.get(cid, 0)
        after = final_class_counts.get(cid, 0)
        gain = after - before
        pct = f"+{(gain/before)*100:.1f}%" if before > 0 else "+0%"
        print(f"[{cid}] {cname:<12} {before:<12} {after:<12} {pct:<10}")
    print(f"\nSaved balanced dataset configuration at: {out_yaml}")
    print("=" * 80 + "\n")
    return str(out_yaml)

def main():
    parser = argparse.ArgumentParser(description="Targeted offline minority class rebalancing for solar datasets")
    parser.add_argument("--source", type=str, default="/home/cse/solar_ai/merged_dataset_v3", help="Source YOLO dataset directory")
    parser.add_argument("--output", type=str, default="/home/cse/solar_ai/merged_dataset_v3_balanced", help="Output balanced dataset directory")
    parser.add_argument("--feather-factor", type=int, default=8, help="Augmentation factor for bird feather (class 5)")
    parser.add_argument("--bird-drop-factor", type=int, default=3, help="Augmentation factor for bird drop (class 2)")
    parser.add_argument("--defect-factor", type=int, default=2, help="Augmentation factor for defect (class 1)")
    parser.add_argument("--clean-factor", type=int, default=2, help="Augmentation factor for clean (class 3)")

    args = parser.parse_args()
    multipliers = {
        5: args.feather_factor,
        2: args.bird_drop_factor,
        1: args.defect_factor,
        3: args.clean_factor
    }
    rebalance_dataset(args.source, args.output, target_multipliers=multipliers)

if __name__ == '__main__':
    main()
