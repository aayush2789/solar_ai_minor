import os
import glob
import random
import cv2
import numpy as np

CLASSES = ['dust', 'defect', 'bird drop', 'snow', 'clean', 'leaf', 'bird feather']

def parse_yolo_label(lbl_path):
    """
    Parses YOLO label text file.
    Returns list of dicts: [{'class': int, 'bbox': [x, y, w, h]}]
    """
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, 'r') as f:
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

def save_yolo_label(lbl_path, boxes):
    """Saves bounding boxes to YOLO label file."""
    with open(lbl_path, 'w') as f:
        for b in boxes:
            f.write(f"{b['class']} {b['bbox'][0]:.6f} {b['bbox'][1]:.6f} {b['bbox'][2]:.6f} {b['bbox'][3]:.6f}\n")

def augment_image_and_boxes(img, boxes, aug_type='flip_h'):
    """
    Applies image augmentations and correctly recalculates normalized YOLO bounding box coordinates.
    aug_type options: 'flip_h', 'color_jitter', 'affine'
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

    elif aug_type == 'color_jitter':
        # HSV Color jitter
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_shift = random.uniform(-10, 10)
        s_scale = random.uniform(0.7, 1.3)
        v_scale = random.uniform(0.7, 1.3)

        hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)

        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        for b in boxes:
            aug_boxes.append({'class': b['class'], 'bbox': list(b['bbox'])})

    elif aug_type == 'affine':
        # Slight random scale and shift
        dx = random.uniform(-0.05, 0.05)
        dy = random.uniform(-0.05, 0.05)
        scale = random.uniform(0.9, 1.1)

        M = np.float32([[scale, 0, dx * w], [0, scale, dy * h]])
        aug_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        for b in boxes:
            cls_id = b['class']
            x, y, bw, bh = b['bbox']
            new_x = np.clip((x * scale) + dx, 0.0, 1.0)
            new_y = np.clip((y * scale) + dy, 0.0, 1.0)
            new_bw = np.clip(bw * scale, 0.0, 1.0)
            new_bh = np.clip(bh * scale, 0.0, 1.0)
            aug_boxes.append({'class': cls_id, 'bbox': [new_x, new_y, new_bw, new_bh]})

    return aug_img, aug_boxes

def rebalance_train_split(img_dir, lbl_dir, target_multipliers=None):
    """
    Scans training set, identifies minority class instances, and generates targeted offline augmentations.
    
    target_multipliers: dict mapping class_id -> number of augmentations per image containing this class
    Default targets:
      6 (bird feather): ~8-10 augmentations per image
      2 (bird drop): ~3 augmentations per image
      1 (defect): ~2 augmentations per image
      3 (snow): ~1-2 augmentations per image
      5 (leaf): ~1 augmentation per image
    """
    if target_multipliers is None:
        target_multipliers = {
            6: 8,  # bird feather (102 -> ~900)
            2: 3,  # bird drop (757 -> ~2200)
            1: 2,  # defect (2040 -> ~4000)
            3: 2,  # snow (1785 -> ~3500)
            5: 1   # leaf (2292 -> ~3400)
        }

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    print(f"\nRebalancing dataset with targeted offline augmentation...")
    print(f"Target multipliers: {target_multipliers}")

    generated_count = 0

    for img_path in img_paths:
        basename = os.path.basename(img_path)
        filename, ext = os.path.splitext(basename)
        lbl_path = os.path.join(lbl_dir, filename + ".txt")

        boxes = parse_yolo_label(lbl_path)
        if not boxes:
            continue

        # Check classes present in this image
        classes_in_img = {b['class'] for b in boxes}
        
        # Determine highest augmentation factor required for this image
        max_augs = 0
        for cls_id in classes_in_img:
            if cls_id in target_multipliers:
                max_augs = max(max_augs, target_multipliers[cls_id])

        if max_augs == 0:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        aug_methods = ['flip_h', 'color_jitter', 'affine']

        for i in range(max_augs):
            aug_type = aug_methods[i % len(aug_methods)]
            aug_img, aug_boxes = augment_image_and_boxes(img, boxes, aug_type=aug_type)

            new_filename = f"{filename}_aug_{aug_type}_{i+1}"
            new_img_path = os.path.join(img_dir, new_filename + ext)
            new_lbl_path = os.path.join(lbl_dir, new_filename + ".txt")

            cv2.imwrite(new_img_path, aug_img)
            save_yolo_label(new_lbl_path, aug_boxes)
            generated_count += 1

    print(f"Offline rebalancing augmentation generated {generated_count} new synthetic minority-class samples.")
    return generated_count
