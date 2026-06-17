"""Prepare calibration dataset for INT8 quantization by cropping faces from CelebA-Spoof."""

import os
import sys
import glob
import cv2
from pathlib import Path

def crop_and_save_face(img_path: str, save_path: str, target_size: int = 128, bbox_expansion_factor: float = 1.5) -> bool:
    img = cv2.imread(img_path)
    if img is None:
        return False

    original_height, original_width = img.shape[:2]

    # Resolve bounding box file
    bbox_file = img_path.replace(".png", "_BB.txt").replace(".jpg", "_BB.txt")
    if not os.path.exists(bbox_file):
        return False

    try:
        with open(bbox_file, "r", encoding="utf-8") as f:
            line = f.readline().strip().split(" ")
            x_ref, y_ref, w_ref, h_ref = map(float, line[:4])

        # Rescale bbox coordinates from 224x224 to original size
        x = int(x_ref * (original_width / 224.0))
        w = int(w_ref * (original_width / 224.0))
        y = int(y_ref * (original_height / 224.0))
        h = int(h_ref * (original_height / 224.0))
    except Exception:
        return False

    # Perform crop with padding
    center_x = x + w // 2
    center_y = y + h // 2
    side_len = int(max(w, h) * bbox_expansion_factor)

    x1 = center_x - side_len // 2
    y1 = center_y - side_len // 2
    x2 = x1 + side_len
    y2 = y1 + side_len

    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - original_height)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - original_width)

    if pad_top or pad_bottom or pad_left or pad_right:
        img = cv2.copyMakeBorder(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )
        x1 += pad_left
        y1 += pad_top
        x2 += pad_left
        y2 += pad_top

    face_crop = img[y1:y2, x1:x2]
    if face_crop.size == 0:
        return False

    # Resize to target size
    h_crop, w_crop = face_crop.shape[:2]
    crop_size = min(h_crop, w_crop)
    interp = cv2.INTER_LANCZOS4 if crop_size < target_size else cv2.INTER_AREA
    final_img = cv2.resize(face_crop, (target_size, target_size), interpolation=interp)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, final_img)
    return True


def main():
    dataset_root = r"C:\Users\ADMIN\Documents\Work\school\DeepL\face-antispoof-onnx-main\face-antispoof-onnx-main\CelebA_Spoof-20260610T183751Z-3-001\CelebA_Spoof\Data\test"
    output_dir = "data_calib"
    
    print(f"Scanning for images in: {dataset_root}")
    
    # Recursively find png files in live and spoof subfolders
    live_images = glob.glob(os.path.join(dataset_root, "**", "live", "*.png"), recursive=True)
    spoof_images = glob.glob(os.path.join(dataset_root, "**", "spoof", "*.png"), recursive=True)
    
    print(f"Found {len(live_images)} live images and {len(spoof_images)} spoof images.")
    
    # Target: 125 live and 125 spoof images
    target_count = 125
    selected_live = live_images[:target_count]
    selected_spoof = spoof_images[:target_count]
    
    success_count = 0
    
    print("\nProcessing live images...")
    for idx, img_path in enumerate(selected_live):
        save_path = os.path.join(output_dir, f"live_{idx:03d}.png")
        if crop_and_save_face(img_path, save_path):
            success_count += 1
            
    print("Processing spoof images...")
    for idx, img_path in enumerate(selected_spoof):
        save_path = os.path.join(output_dir, f"spoof_{idx:03d}.png")
        if crop_and_save_face(img_path, save_path):
            success_count += 1
            
    print(f"\nDone! Preprocessed and saved {success_count} calibration images to: {output_dir}")


if __name__ == "__main__":
    main()
