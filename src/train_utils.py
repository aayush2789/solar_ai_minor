import os
import yaml
import torch

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

def load_yaml_config(yaml_path):
    """Loads YAML configuration file."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def calculate_class_weights(class_counts, total_samples, num_classes=7):
    """
    Calculates inverse class weights for loss weighting or sampling.
    weights[i] = total_samples / (num_classes * count[i])
    """
    weights = {}
    for cid in range(num_classes):
        cnt = class_counts.get(cid, 1)
        if cnt == 0:
            cnt = 1
        weights[cid] = round(total_samples / (num_classes * cnt), 4)
    return weights
