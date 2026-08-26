
import argparse
from collections import defaultdict

def analyze_labels(dataset_dir):
    splits = ['train', 'valid', 'test']
    
    print(f"Analyzing dataset at: {os.path.abspath(dataset_dir)}\n")
    
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        labels_dir = os.path.join(split_dir, "labels")
        
        if not os.path.exists(labels_dir):
            print(f"Split '{split}': No labels directory found at {labels_dir}")
            continue
            
        stats = {
            'total_files': 0,
            'clean_detection': 0,
            'clean_segmentation': 0,
            'mixed': 0,
            'empty': 0,
            'other': 0
        }
        mixed_examples = []
        clean_segmentation_examples = []
        
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        stats['total_files'] = len(label_files)
        
        for fname in label_files:
            fpath = os.path.join(labels_dir, fname)
            
            det_count = 0
            seg_count = 0
            other_count = 0
            
            try:
                with open(fpath, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    
                if not lines:
                    stats['empty'] += 1
                    continue
                    
                for line in lines:
                    parts = line.split()
                    num_elements = len(parts)
                    
                    if num_elements == 5:
                        det_count += 1
                    elif num_elements > 5:
                        seg_count += 1
                    else:
                        other_count += 1
                        
                if det_count > 0 and seg_count > 0:
                    stats['mixed'] += 1
                    mixed_examples.append(fname)
                elif det_count > 0 and seg_count == 0 and other_count == 0:
                    stats['clean_detection'] += 1
                elif seg_count > 0 and det_count == 0 and other_count == 0:
                    stats['clean_segmentation'] += 1
                    clean_segmentation_examples.append(fname)
                elif other_count > 0 and det_count == 0 and seg_count == 0:
                    stats['other'] += 1
                else:
                    # e.g., combinations with other_count > 0
                    stats['mixed'] += 1
                    mixed_examples.append(fname)
                    
            except Exception as e:
                print(f"Error reading {fname}: {e}")
                
        print(f"Split: {split}")
        print(f"  Total label files found: {stats['total_files']}")
        print(f"  Clean Detection files (only 5-col rows): {stats['clean_detection']}")
        print(f"  Clean Segmentation files (only >5-col rows): {stats['clean_segmentation']}")
        print(f"  Mixed files (mix of detection & segment rows): {stats['mixed']}")
        print(f"  Empty files: {stats['empty']}")
        print(f"  Other invalid files: {stats['other']}")
        print()
        
        if stats['mixed'] > 0:
            print(f"  First 5 mixed files in {split}:")
            for ex in mixed_examples[:5]:
                print(f"    - {ex}")
            print()
            
        if stats['clean_segmentation'] > 0:
            print(f"  First 5 clean segmentation files in {split}:")
            for ex in clean_segmentation_examples[:5]:
                print(f"    - {ex}")
            print()
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze YOLO labels for detection/segmentation mixing.")
    parser.add_argument("--dir", type=str, default="/home/cse/solar_ai/dataset", help="Path to dataset directory")
    args = parser.parse_args()
    
    analyze_labels(args.dir)
