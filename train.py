import os
import argparse
import yaml
from ultralytics import YOLO
from src.train_utils import check_environment, load_yaml_config

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

def setup_wandb(args):
    """Configures Weights & Biases experiment tracking."""
    if not args.wandb:
        # Disable wandb in ultralytics settings if requested
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
        print("Note: WANDB_API_KEY not found in environment/.env. W&B will run in anonymous or default logged-in mode.")

    print(f"W&B Tracking Enabled -> Project: '{project_name}', Run Name: '{run_name}'")
    return True

def train(args):
    """Executes YOLOv11 training pipeline with custom hyperparameter configuration and W&B logging."""
    check_environment()
    use_wandb = setup_wandb(args)

    # Verify dataset configuration exists
    data_yaml = args.data
    if not os.path.exists(data_yaml):
        alt_yaml = os.path.join(args.dataset_dir, "data.yaml")
        if os.path.exists(alt_yaml):
            data_yaml = alt_yaml
        else:
            raise FileNotFoundError(f"Dataset configuration not found at {data_yaml} or {alt_yaml}. Please run preprocessing first.")

    # Load custom hyperparameter config
    hyp_dict = {}
    if os.path.exists(args.hyp):
        raw_hyp = load_yaml_config(args.hyp)
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
            hyp_dict = {k: v for k, v in raw_hyp.items() if k in DEFAULT_CFG_DICT}
            ignored = set(raw_hyp.keys()) - set(hyp_dict.keys())
            if ignored:
                print(f"Ignored non-standard/deprecated hyperparameter keys: {ignored}")
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
    print(f"W&B Tracking       : {'Enabled' if use_wandb else 'Disabled'}")
    print(f"Project Output Dir : {args.project}/{args.name}")
    print(f"===========================================================\n")

    # Initialize YOLO Model (yolo11n.pt / yolo11s.pt)
    model = YOLO(args.model)

    # Combine CLI parameters with YAML hyperparameter parameters
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

    # Merge custom hyperparameters (fl_gamma, cls, mosaic, mixup, etc.)
    train_params.update(hyp_dict)

    # Start Training
    results = model.train(**train_params)

    print("\n================ TRAINING FINISHED ================")
    save_dir = getattr(results, 'save_dir', os.path.join(args.project, args.name))
    print(f"Model checkpoints and logs saved at: {save_dir}")
    print(f"Best model weights: {os.path.join(save_dir, 'weights', 'best.pt')}")

    if use_wandb and WANDB_AVAILABLE:
        try:
            best_weights_path = os.path.join(save_dir, 'weights', 'best.pt')
            if os.path.exists(best_weights_path) and wandb.run is not None:
                artifact = wandb.Artifact(f"{args.name}-best-model", type="model")
                artifact.add_file(best_weights_path)
                wandb.log_artifact(artifact)
                print("Uploaded best model weights artifact to W&B.")
        except Exception as e:
            print(f"Could not upload W&B artifact: {e}")

    print("===================================================\n")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv11 Training Pipeline for Solar Panel Defect Detection")
    parser.add_argument("--model", type=str, default="yolo11s.pt", choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt"], help="YOLOv11 model variant")
    parser.add_argument("--data", type=str, default="/home/cse/solar_ai/configs/data.yaml", help="Path to dataset data.yaml")
    parser.add_argument("--dataset-dir", type=str, default="/home/cse/solar_ai/dataset", help="Root dataset directory")
    parser.add_argument("--hyp", type=str, default="/home/cse/solar_ai/configs/train_config.yaml", help="Path to train hyperparameter YAML")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="0", help="GPU device ID (e.g. '0' or 'cpu')")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience epochs")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--project", type=str, default="/home/cse/solar_ai/runs/detect", help="Project directory to save runs")
    parser.add_argument("--name", type=str, default="solar_yolo11s", help="Run experiment name")
    parser.add_argument("--wandb", action="store_true", default=True, help="Enable Weights & Biases experiment tracking")
    parser.add_argument("--no-wandb", action="store_false", dest="wandb", help="Disable Weights & Biases experiment tracking")
    parser.add_argument("--wandb-project", type=str, default="solar_panel_defect_detection", help="W&B Project Name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B Entity/Username")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B Run Name")

    args = parser.parse_args()
    train(args)
