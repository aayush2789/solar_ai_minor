import os
import argparse
import glob
from ultralytics import YOLO

def predict(args):
    """Runs YOLO inference on target image or image directory."""
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Model weights file not found at: {args.weights}")

    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Inference source not found at: {args.source}")

    print(f"\n================ INFERENCE PIPELINE ================")
    print(f"Model Weights  : {args.weights}")
    print(f"Inference Source: {args.source}")
    print(f"Confidence Thresh: {args.conf}")
    print(f"IoU Thresh       : {args.iou}")
    print(f"Device           : {args.device}")
    print(f"===================================================\n")

    model = YOLO(args.weights)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        save_txt=True,
        save_conf=True,
        project=args.project,
        name=args.name,
        exist_ok=True
    )

    out_dir = os.path.join(args.project, args.name)
    print(f"\nInference completed! Predictions saved to: {out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run YOLOv11 Model Inference")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights (.pt)")
    parser.add_argument("--source", type=str, default="/home/cse/solar_ai/dataset/test/images", help="Path to input image file or directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="0", help="GPU device ID")
    parser.add_argument("--project", type=str, default="/home/cse/solar_ai/runs/predict", help="Prediction output directory")
    parser.add_argument("--name", type=str, default="exp", help="Run experiment name")

    args = parser.parse_args()
    predict(args)
