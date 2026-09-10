import os
import glob
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional

import yaml
import torch

DEFAULT_CLASS_NAMES = [
    'dust',          # 0
    'defect',        # 1
    'bird drop',     # 2
    'snow',          # 3
    'clean',         # 4
    'leaf',          # 5
    'bird feather'   # 6
]

def check_environment():
    """Prints system environment specs and GPU details."""
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    
    print(f"--- Environment Check ---")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {cuda_available}")
    print(f"Device Count: {device_count}")
    print(f"Primary Device: {device_name}")
    print(f"--------------------------")
    return cuda_available, device_name

def load_yaml_config(yaml_path: str) -> dict:
    """Loads YAML configuration file."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_train_class_counts(data_yaml_path: str) -> Tuple[Dict[int, int], int, Dict[int, int], List[str]]:
    """
    Parses the dataset data.yaml, resolves the training labels directory,
    and calculates bounding box counts per class and image occurrence counts.
    Returns (box_counts, total_boxes, image_counts, class_names).
    """
    data_cfg = load_yaml_config(data_yaml_path)
    base_path = data_cfg.get('path', os.path.dirname(data_yaml_path))
    train_split = data_cfg.get('train', 'train/images')

    # Resolve train image and label directories
    if os.path.isabs(train_split):
        train_img_dir = train_split
    else:
        train_img_dir = os.path.join(base_path, train_split)

    train_lbl_dir = train_img_dir.replace('images', 'labels')
    if not os.path.isdir(train_lbl_dir):
        # Alternative fallback
        train_lbl_dir = os.path.join(os.path.dirname(train_img_dir), 'labels')

    names_entry = data_cfg.get('names', DEFAULT_CLASS_NAMES)
    if isinstance(names_entry, dict):
        class_names = [names_entry[i] for i in range(len(names_entry))]
    else:
        class_names = list(names_entry)

    box_counts = Counter()
    image_counts = Counter()
    total_boxes = 0

    if os.path.isdir(train_lbl_dir):
        label_files = glob.glob(os.path.join(train_lbl_dir, "*.txt"))
        for lf in label_files:
            try:
                with open(lf, 'r', encoding='utf-8') as f:
                    classes_in_file = set()
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                cid = int(parts[0])
                                box_counts[cid] += 1
                                total_boxes += 1
                                classes_in_file.add(cid)
                            except ValueError:
                                continue
                    for c in classes_in_file:
                        image_counts[c] += 1
            except Exception:
                continue

    return dict(box_counts), total_boxes, dict(image_counts), class_names

def calculate_class_weights(
    class_counts: Dict[int, int],
    num_classes: int = 7,
    method: str = "smoothed",
    alpha: float = 0.4,
    beta: float = 0.999
) -> Dict[int, float]:
    """
    Calculates balanced class weights to address representation disparity in loss functions.
    
    Methods:
      - 'smoothed': (N_max / n_i) ** alpha (stable, recommended for detection)
      - 'effective': Class-Balanced Loss (Cui et al., CVPR 2019) (1 - beta) / (1 - beta ** n_i)
      - 'sqrt': (N_max / n_i) ** 0.5
      - 'inverse': Total / (num_classes * n_i)
      - 'none': Uniform 1.0 weights
    """
    if method == "none":
        return {cid: 1.0 for cid in range(num_classes)}

    counts = [max(1, class_counts.get(cid, 1)) for cid in range(num_classes)]
    max_count = max(counts)
    total_samples = sum(counts)

    weights = {}

    if method == "smoothed":
        # Power-law smoothed inverse frequency
        for cid in range(num_classes):
            w = (max_count / counts[cid]) ** alpha
            weights[cid] = round(float(w), 4)

    elif method == "effective":
        # Effective number of samples (Cui et al.)
        for cid in range(num_classes):
            effective_num = (1.0 - math.pow(beta, counts[cid])) / (1.0 - beta)
            w = 1.0 / max(1e-6, effective_num)
            weights[cid] = w
        # Normalize so minimum weight is 1.0
        min_w = min(weights.values())
        weights = {cid: round(float(w / min_w), 4) for cid, w in weights.items()}

    elif method == "sqrt":
        for cid in range(num_classes):
            w = math.sqrt(max_count / counts[cid])
            weights[cid] = round(float(w), 4)

    elif method == "inverse":
        for cid in range(num_classes):
            w = total_samples / (num_classes * counts[cid])
            weights[cid] = round(float(w), 4)

    else:
        raise ValueError(f"Unknown class weighting method: {method}")

    return weights

def print_imbalance_report(
    class_counts: Dict[int, int],
    class_names: List[str],
    class_weights: Dict[int, float]
):
    """Prints a clear terminal overview of the training class balance and computed loss weights."""
    total_boxes = sum(class_counts.values()) if class_counts else 1

    print("\n" + "=" * 80)
    print("           TRAINING DATASET CLASS IMBALANCE & LOSS WEIGHTS")
    print("=" * 80)
    print(f"{'Class ID':<9} {'Class Name':<15} {'Box Count':<12} {'Percentage':<12} {'Loss Weight':<12}")
    print("-" * 80)

    for cid in range(len(class_names)):
        cname = class_names[cid]
        cnt = class_counts.get(cid, 0)
        pct = (cnt / total_boxes) * 100.0 if total_boxes > 0 else 0.0
        w = class_weights.get(cid, 1.0)
        print(f"[{cid}]      {cname:<15} {cnt:<12} {pct:>6.2f}%      {w:>8.3f}x")

    print("=" * 80 + "\n")

def register_class_weights_callback(model, class_weights: Dict[int, float]):
    """
    Registers an Ultralytics YOLO callback to inject class_weights into
    the model and its criterion (v8DetectionLoss) during training.
    """
    num_classes = len(class_weights)
    weights_list = [class_weights.get(i, 1.0) for i in range(num_classes)]
    weights_tensor = torch.tensor(weights_list, dtype=torch.float32)

    def set_weights_hook(trainer):
        device = trainer.device
        w_tensor = weights_tensor.to(device)

        if hasattr(trainer, 'model'):
            trainer.model.class_weights = w_tensor
            if hasattr(trainer.model, 'criterion') and trainer.model.criterion is not None:
                trainer.model.criterion.class_weights = w_tensor.view(1, 1, -1)

    model.add_callback('on_pretrain_routine_end', set_weights_hook)
    model.add_callback('on_train_start', set_weights_hook)
    return weights_tensor
