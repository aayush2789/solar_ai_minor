import os
import shutil
import argparse
import sys

# Add parent directory to path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.deduplicate import remove_duplicates
from preprocessing.rebalance import rebalance_train_split
from src.dataset_utils import print_dataset_report

def run_pipeline(source_dir, output_dir):
    """
    Executes master preprocessing pipeline:
      1. Copy and deduplicate raw dataset into target directory.
      2. Rebalance training split with targeted offline augmentations for minority classes.
      3. Create dataset data.yaml file.
      4. Print full class distribution report.
    """
    print(f"\n================ STARTING PREPROCESSING PIPELINE ================")
    print(f"Source Raw Dataset: {source_dir}")
    print(f"Target Processed Dataset: {output_dir}\n")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Process Splits with Deduplication
    splits = ['train', 'valid', 'test']
    for split in splits:
        src_img = os.path.join(source_dir, split, "images")
        src_lbl = os.path.join(source_dir, split, "labels")

        out_img = os.path.join(output_dir, split, "images")
        out_lbl = os.path.join(output_dir, split, "labels")

        print(f"\n--- Deduplicating split: {split} ---")
        if os.path.exists(src_img):
            # Apply deduplication threshold (hash_threshold=2 for train, 1 for valid/test)
            thresh = 2 if split == 'train' else 1
            remove_duplicates(src_img, src_lbl, out_img, out_lbl, hash_threshold=thresh)

    # 2. Rebalance Training Split
    print("\n--- Rebalancing Training Split ---")
    train_img = os.path.join(output_dir, "train", "images")
    train_lbl = os.path.join(output_dir, "train", "labels")
    
    rebalance_train_split(train_img, train_lbl)

    # 3. Create dataset YAML
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: train/images
val: valid/images
test: test/images

nc: 7
names: ['dust', 'defect', 'bird drop', 'snow', 'clean', 'leaf', 'bird feather']
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nDataset config created at: {yaml_path}")

    # 4. Final Class Distribution Summary Report
    print("\nFinal Preprocessed Dataset Summary:")
    print_dataset_report(output_dir)
    print("================ PREPROCESSING COMPLETED SUCCESSFULLY ================\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Master Data Preprocessing & Rebalancing Pipeline")
    parser.add_argument("--source", type=str, default="/home/cse/solar_ai/solar_datasets/merged_dataset_v2", help="Source raw merged dataset")
    parser.add_argument("--output", type=str, default="/home/cse/solar_ai/dataset", help="Target output dataset directory")
    args = parser.parse_args()

    run_pipeline(args.source, args.output)
