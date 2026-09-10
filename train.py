#!/usr/bin/env python3
"""
Enhanced YOLOv11 Training Pipeline for Solar Panel Defect Detection
Optimized for severe class imbalance in merged_dataset_v3.

Features:
- Automatic training set class distribution inspection
- Dynamic class weight calculation (smoothed power-inverse, effective samples)
- Native loss weighting integration via Ultralytics callbacks
- Targeted online and optional offline minority oversampling
- Comprehensive per-class evaluation metrics reporting
- W&B experiment tracking integration
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional

import yaml
import torch
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT

from src.train_utils import (
    check_environment,
    load_yaml_config,
    get_train_class_counts,
    calculate_class_weights,
    print_imbalance_report,
    register_class_weights_callback
)
from preprocessing.rebalance import rebalance_dataset

# Attempt to load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Attempt to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_wandb(args) -> bool:
    """Configures Weights & Biases experiment tracking."""
    if not args.wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return False

    if not WANDB_AVAILABLE:
        print("Warning: wandb package is not installed. Running without W&B tracking.")
        os.environ["WANDB_DISABLED"] = "true"
        return False

    project_name = args.wandb_project or os.getenv("WANDB_PROJECT", "solar_panel_defect_detection")
    entity_name = args.wandb_entity or os.getenv("WANDB_ENTITY", None)
    run_name = args.wandb_name or args.name

    os.environ["WANDB_PROJECT"] = project_name
    if entity_name:
        os.environ["WANDB_ENTITY"] = entity_name
    if run_name:
        os.environ["WANDB_NAME"] = run_name
    os.environ.pop("WANDB_DISABLED", None)

    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        try:
            wandb.login(key=api_key)
            print("Successfully authenticated with W&B using WANDB_API_KEY.")
        except Exception as e:
            print(f"W&B authentication error: {e}")
    else:
        print("Note: WANDB_API_KEY not found in environment/.env. Running with standard local logs.")

    print(f"W&B Tracking Enabled -> Project: '{project_name}', Run Name: '{run_name}'")
    return True


def train(args):
    """Executes YOLOv11 training pipeline with imbalance-aware loss weighting and augmentations."""
    check_environment()
    use_wandb = setup_wandb(args)

    # 1. Resolve Dataset YAML Path
    data_yaml = args.data
    if not os.path.exists(data_yaml):
        fallbacks = [
            os.path.join(args.dataset_dir, "data.yaml"),
            "/home/cse/solar_ai/merged_dataset_v3/data.yaml",
            "/home/cse/solar_ai/configs/data.yaml"
        ]
        found = False
        for fb in fallbacks:
            if os.path.exists(fb):
                data_yaml = fb
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Dataset configuration not found at {data_yaml} or fallbacks.")

    print(f"Using Dataset Configuration: {data_yaml}")

    # 2. Optional Offline Rebalancing
    if args.rebalance:
        print("\n[INFO] Offline rebalancing requested before training.")
        source_dir = os.path.dirname(data_yaml) if os.path.isabs(data_yaml) else "/home/cse/solar_ai/merged_dataset_v3"
        output_dir = args.balanced_dir or "/home/cse/solar_ai/merged_dataset_v3_balanced"
        data_yaml = rebalance_dataset(source_dir, output_dir)

    # 3. Analyze Class Distribution in Training Set
    class_counts, total_boxes, img_counts, class_names = get_train_class_counts(data_yaml)
    num_classes = len(class_names)

    # 4. Compute Loss Class Weights
    weight_method = "smoothed" if args.class_weights == "auto" else args.class_weights
    class_weights = calculate_class_weights(
        class_counts=class_counts,
        num_classes=num_classes,
        method=weight_method,
        alpha=args.weight_alpha,
        beta=args.weight_beta
    )

    # Display Imbalance & Weighting Report
    print_imbalance_report(class_counts, class_names, class_weights)

    # 5. Load Custom Hyperparameter Config
    hyp_dict = {}
    if os.path.exists(args.hyp):
        raw_hyp = load_yaml_config(args.hyp)
        try:
            hyp_dict = {k: v for k, v in raw_hyp.items() if k in DEFAULT_CFG_DICT}
            ignored = set(raw_hyp.keys()) - set(hyp_dict.keys())
            if ignored:
                print(f"Ignored non-standard hyperparameter keys: {ignored}")
        except Exception:
            hyp_dict = raw_hyp
        print(f"Loaded custom hyperparameter configuration from {args.hyp}")

    print(f"\n================ YOLOv11 TRAINING PIPELINE ================")
    print(f"Model Architecture : {args.model}")
    print(f"Dataset Config     : {data_yaml}")
    print(f"Epochs             : {args.epochs}")
    print(f"Batch Size         : {args.batch}")
    print(f"Image Resolution   : {args.imgsz}")
    print(f"Device             : {args.device}")
    print(f"Class Weighting    : {args.class_weights} (method: {weight_method})")
    print(f"Classification Gain: {hyp_dict.get('cls', 0.5)}")
    print(f"Copy-Paste Aug     : {hyp_dict.get('copy_paste', 0.0)}")
    print(f"W&B Tracking       : {'Enabled' if use_wandb else 'Disabled'}")
    print(f"Project Output Dir : {args.project}/{args.name}")
    print(f"===========================================================\n")

    # 6. Initialize YOLO Model
    model = YOLO(args.model)

    # 7. Inject Class Weights via Ultralytics Callback Hook
    if args.class_weights != "none":
        register_class_weights_callback(model, class_weights)
        print("Successfully attached class-weighting callback to model training loss!")

    # 8. Assemble Training Parameters
    train_params = {
        'data': data_yaml,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'project': args.project,
        'name': args.name,
        'exist_ok': True,
        'save': True,
        'plots': True,
        'patience': args.patience,
        'workers': args.workers
    }
    train_params.update(hyp_dict)

    # 9. Start Training
    results = model.train(**train_params)

    save_dir = getattr(results, 'save_dir', os.path.join(args.project, args.name))
    best_weights_path = os.path.join(save_dir, 'weights', 'best.pt')

    print("\n================ TRAINING FINISHED ================")
    print(f"Model checkpoints and logs saved at: {save_dir}")
    print(f"Best model weights: {best_weights_path}")

    # 10. Post-Training Per-Class Evaluation
    if args.evaluate and os.path.exists(best_weights_path):
        print("\nRunning comprehensive per-class validation evaluation...")
        best_model = YOLO(best_weights_path)
        val_results = best_model.val(data=data_yaml, split='val', device=args.device)

        print("\n" + "=" * 80)
        print("                 PER-CLASS VALIDATION PERFORMANCE")
        print("=" * 80)
        print(f"{'Class Name':<16} {'mAP50':<12} {'mAP50-95':<12} {'Precision':<12} {'Recall':<12}")
        print("-" * 80)

        # Extract per class metrics if available
        try:
            maps = val_results.box.maps  # array of mAP50-95 per class
            map50s = val_results.box.ap50  # array of mAP50 per class
            precisions = val_results.box.p
            recalls = val_results.box.r

            for cid, cname in enumerate(class_names):
                m50 = map50s[cid] if cid < len(map50s) else 0.0
                m = maps[cid] if cid < len(maps) else 0.0
                p = precisions[cid] if cid < len(precisions) else 0.0
                r = recalls[cid] if cid < len(recalls) else 0.0
                print(f"{cname:<16} {m50:<12.4f} {m:<12.4f} {p:<12.4f} {r:<12.4f}")
        except Exception as e:
            print(f"Summary metrics: mAP50: {val_results.box.map50:.4f}, mAP50-95: {val_results.box.map:.4f}")
        print("=" * 80 + "\n")

    # W&B Artifact Upload
    if use_wandb and WANDB_AVAILABLE:
        try:
            if os.path.exists(best_weights_path) and wandb.run is not None:
                artifact = wandb.Artifact(f"{args.name}-best-model", type="model")
                artifact.add_file(best_weights_path)
                wandb.log_artifact(artifact)
                print("Uploaded best model weights artifact to W&B.")
        except Exception as e:
            print(f"Could not upload W&B artifact: {e}")

    print("===================================================\n")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Imbalance-Aware YOLOv11 Training Pipeline for Solar Panel Defect Detection")
    parser.add_argument("--model", type=str, default="yolo11s.pt", choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt"], help="YOLOv11 model variant")
    parser.add_argument("--data", type=str, default="/home/cse/solar_ai/merged_dataset_v3/data.yaml", help="Path to dataset data.yaml")
    parser.add_argument("--dataset-dir", type=str, default="/home/cse/solar_ai/merged_dataset_v3", help="Root dataset directory")
    parser.add_argument("--hyp", type=str, default="/home/cse/solar_ai/configs/train_config.yaml", help="Path to train hyperparameter YAML")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="0", help="GPU device ID (e.g. '0' or 'cpu')")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience epochs")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--project", type=str, default="/home/cse/solar_ai/runs/detect", help="Project directory to save runs")
    parser.add_argument("--name", type=str, default="solar_yolo11s_imbalanced", help="Run experiment name")

    # Imbalance handling arguments
    parser.add_argument("--class-weights", type=str, default="auto", choices=["auto", "smoothed", "effective", "sqrt", "inverse", "none"], help="Class weighting method for detection loss")
    parser.add_argument("--weight-alpha", type=float, default=0.4, help="Power exponent for smoothed class weights (default: 0.4)")
    parser.add_argument("--weight-beta", type=float, default=0.999, help="Beta parameter for effective number of samples weighting")
    parser.add_argument("--rebalance", action="store_true", default=False, help="Perform targeted offline augmentation on minority classes before training")
    parser.add_argument("--balanced-dir", type=str, default=None, help="Directory to store offline rebalanced dataset")
    parser.add_argument("--evaluate", action="store_true", default=True, help="Run validation evaluation and print per-class metrics after training")

    # W&B arguments
    parser.add_argument("--wandb", action="store_true", default=True, help="Enable Weights & Biases experiment tracking")
    parser.add_argument("--no-wandb", action="store_false", dest="wandb", help="Disable Weights & Biases experiment tracking")
    parser.add_argument("--wandb-project", type=str, default="solar_panel_defect_detection", help="W&B Project Name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B Entity/Username")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B Run Name")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
