import os
import glob
import hashlib
import cv2
import numpy as np
from PIL import Image

def compute_md5(file_path):
    """Computes exact MD5 checksum of a file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def compute_dhash(img, hash_size=8):
    """
    Computes difference hash (dhash) for perceptual similarity comparison.
    img: numpy array (BGR image) or PIL Image
    Returns integer hash value.
    """
    if isinstance(img, np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
    else:
        gray = img.convert('L')
        resized = gray.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        resized = np.array(resized)

    # Compute difference between adjacent pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # Convert binary matrix to bit integer
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def hamming_distance(h1, h2):
    """Calculates bitwise Hamming distance between two 64-bit integers."""
    return bin(h1 ^ h2).count('1')

def remove_duplicates(images_dir, labels_dir, output_img_dir, output_lbl_dir, hash_threshold=2):
    """
    Scans image directory, computes MD5 and perceptual dhashes, removes duplicate images/labels,
    and copies unique pairs to output directories.
    
    Returns:
        dict: {'total': int, 'unique': int, 'duplicates_md5': int, 'duplicates_phash': int}
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    seen_md5 = set()
    seen_dhashes = [] # list of (dhash, img_basename)

    stats = {
        'total': len(img_paths),
        'unique': 0,
        'duplicates_md5': 0,
        'duplicates_phash': 0
    }

    print(f"Scanning {len(img_paths)} images for duplicates...")

    for img_path in img_paths:
        basename = os.path.basename(img_path)
        filename, ext = os.path.splitext(basename)
        lbl_path = os.path.join(labels_dir, filename + ".txt")

        # 1. Exact MD5 hash check
        md5_val = compute_md5(img_path)
        if md5_val in seen_md5:
            stats['duplicates_md5'] += 1
            continue

        # 2. Perceptual hash check (dhash)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Warning: Unable to read image {img_path}")
            continue

        dhash_val = compute_dhash(img)
        is_near_duplicate = False

        for prev_dhash, prev_name in seen_dhashes:
            if hamming_distance(dhash_val, prev_dhash) <= hash_threshold:
                is_near_duplicate = True
                stats['duplicates_phash'] += 1
                break

        if is_near_duplicate:
            continue

        # Keep image
        seen_md5.add(md5_val)
        seen_dhashes.append((dhash_val, basename))

        out_img_path = os.path.join(output_img_dir, basename)
        out_lbl_path = os.path.join(output_lbl_dir, filename + ".txt")

        cv2.imwrite(out_img_path, img)

        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f_in, open(out_lbl_path, 'w') as f_out:
                f_out.write(f_in.read())
        else:
            open(out_lbl_path, 'w').close()

        stats['unique'] += 1

    print(f"Deduplication complete: {stats['total']} total -> {stats['unique']} unique ({stats['duplicates_md5']} exact MD5 dupes, {stats['duplicates_phash']} near-perceptual dupes removed).")
    return stats
