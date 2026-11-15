import os
import argparse
from ultralytics import YOLO

CLASSES = ['dust', 'defect', 'bird drop', 'clean', 'leaf', 'bird feather']

def evaluate(args):
    """Evaluates trained YOLO model on validation or test split."""
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Model weights file not found at: {args.weights}")

    print(f"\n================ EVALUATION PIPELINE ================")
    print(f"Model Weights  : {args.weights}")
    print(f"Data Config    : {args.data}")
    print(f"Split          : {args.split}")
    print(f"Image Size     : {args.imgsz}")
    print(f"Device         : {args.device}")
    print(f"=====================================================\n")

    model = YOLO(args.weights)
    
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True
    )

    print("\n---------------- EVALUATION SUMMARY ----------------")
    print(f"Overall mAP50    : {metrics.box.map50:.4f}")
    print(f"Overall mAP50-95 : {metrics.box.map:.4f}")
    print(f"Overall Precision: {metrics.box.mp:.4f}")
    print(f"Overall Recall   : {metrics.box.mr:.4f}\n")

    print("Per-Class Metrics breakdown:")
    print(f"{'Class ID':<10} {'Class Name':<15} {'Precision':<12} {'Recall':<12} {'mAP50':<12} {'mAP50-95':<12}")
    print("-" * 75)

    names = model.names if hasattr(model, 'names') and model.names else {i: c for i, c in enumerate(CLASSES)}
    class_names = [names[i] for i in range(len(names))]
    maps_per_class = metrics.box.maps
    for cid, cname in enumerate(class_names):
        p = metrics.box.p[cid] if cid < len(metrics.box.p) else 0.0
        r = metrics.box.r[cid] if cid < len(metrics.box.r) else 0.0
        map50 = maps_per_class[cid] if cid < len(maps_per_class) else 0.0
        print(f"{cid:<10} {cname:<15} {p:<12.4f} {r:<12.4f} {map50:<12.4f}")

    print("=====================================================\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11 Model Performance")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights (.pt)")
    parser.add_argument("--data", type=str, default="/home/cse/solar_ai/configs/data.yaml", help="Path to dataset data.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["val", "valid", "test"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="GPU device ID")
    parser.add_argument("--project", type=str, default="/home/cse/solar_ai/runs/eval", help="Evaluation output directory")
    parser.add_argument("--name", type=str, default="test_eval", help="Run experiment name")

    args = parser.parse_args()
    evaluate(args)
