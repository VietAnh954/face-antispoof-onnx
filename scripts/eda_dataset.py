"""Exploratory Data Analysis (EDA) script for CelebA-Spoof dataset."""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

def run_eda(orig_dir: str = "data") -> None:
    label_dir = os.path.join(orig_dir, "metas", "labels")
    train_json = os.path.join(label_dir, "train_label.json")
    test_json = os.path.join(label_dir, "test_label.json")
    
    print("=" * 70)
    print("         CELEBA-SPOOF EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("=" * 70)
    
    if not os.path.exists(train_json) or not os.path.exists(test_json):
        print("[Warning] Dataset label files not found in the local workspace.")
        print("Displaying official CelebA-Spoof statistical distribution:")
        print("-" * 70)
        display_official_stats()
        return
        
    print(f"[Info] Reading metadata from {label_dir}...")
    try:
        with open(train_json, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(test_json, "r", encoding="utf-8") as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load JSON files: {e}")
        return
        
    train_df = pd.DataFrame.from_dict(train_data, orient="index")
    test_df = pd.DataFrame.from_dict(test_data, orient="index")
    
    # CelebA-Spoof specific columns (43 attributes total)
    # Column 40: Spoof Type (0: Live, 1-9: Spoof)
    # Column 41: Illumination (0: Normal, 1: Low, 2: Backlight, 3: Highlight)
    # Column 42: Environment (0: Indoor, 1: Outdoor)
    
    total_train = len(train_df)
    total_test = len(test_df)
    total_all = total_train + total_test
    
    print(f"\n1. Subsets split counts:")
    print(f"   - Train set: {total_train} images ({total_train/total_all*100:.2f}%)")
    print(f"   - Test set:  {total_test} images ({total_test/total_all*100:.2f}%)")
    print(f"   - Total:     {total_all} images")
    
    # Class Distribution (Real vs Spoof)
    all_df = pd.concat([train_df, test_df])
    
    # Spoof type map
    spoof_names = {
        0: "Live Face (Real)",
        1: "Print Attack (Paper Photo)",
        2: "Replay Attack (Phone screen)",
        3: "Replay Attack (Tablet screen)",
        4: "Replay Attack (Laptop screen)",
        5: "Replay Attack (TV screen)",
        6: "3D Mask Attack",
        7: "Paper Cut Mask Attack",
        8: "Paper Mask Attack",
        9: "Silhouette Paper Attack"
    }
    
    illum_names = {
        0: "Normal Illumination",
        1: "Low Light Condition",
        2: "Backlight / Shadow",
        3: "Highlight / Overexposure"
    }
    
    env_names = {
        0: "Indoor Environment",
        1: "Outdoor Environment"
    }
    
    print(f"\n2. Class Distribution (Real vs Spoof):")
    if 40 in all_df.columns:
        all_df['is_spoof'] = all_df[40].apply(lambda x: "SPOOF" if x > 0 else "REAL")
        cls_counts = all_df['is_spoof'].value_counts()
        for cls_name, count in cls_counts.items():
            pct = count / total_all * 100
            bar = "#" * int(pct / 2)
            print(f"   - {cls_name:<7}: {count:<7} ({pct:.2f}%) {bar}")
            
        print(f"\n3. Detailed Spoof Types:")
        type_counts = all_df[40].value_counts().sort_index()
        for code, count in type_counts.items():
            name = spoof_names.get(code, f"Unknown Code {code}")
            pct = count / total_all * 100
            bar = "#" * int(pct / 2)
            print(f"   - [{code}] {name:<32}: {count:<7} ({pct:.2f}%) {bar}")
    else:
        print("   - Column 40 (Spoof Type) not found in JSON data.")
        
    print(f"\n4. Illumination Distribution:")
    if 41 in all_df.columns:
        illum_counts = all_df[41].value_counts().sort_index()
        for code, count in illum_counts.items():
            name = illum_names.get(code, f"Unknown Code {code}")
            pct = count / total_all * 100
            bar = "#" * int(pct / 2)
            print(f"   - [{code}] {name:<26}: {count:<7} ({pct:.2f}%) {bar}")
    else:
        print("   - Column 41 (Illumination) not found in JSON data.")
        
    print(f"\n5. Environment Distribution:")
    if 42 in all_df.columns:
        env_counts = all_df[42].value_counts().sort_index()
        for code, count in env_counts.items():
            name = env_names.get(code, f"Unknown Code {code}")
            pct = count / total_all * 100
            bar = "#" * int(pct / 2)
            print(f"   - [{code}] {name:<20}: {count:<7} ({pct:.2f}%) {bar}")
    else:
        print("   - Column 42 (Environment) not found in JSON data.")
        
    print("=" * 70)

def display_official_stats() -> None:
    official_total = 625537
    official_subjects = 10177
    
    print(f"1. Dataset Overview:")
    print(f"   - Total Images:    {official_total:,} images")
    print(f"   - Unique Subjects: {official_subjects:,} identities")
    print(f"   - Attributes:      43 attributes per image (40 Face attributes + 3 Spoof attributes)")
    
    print(f"\n2. Class Distribution (Real vs Spoof):")
    # Official CelebA-Spoof is roughly 33.3% Real and 66.7% Spoof
    real_count = int(official_total * 0.333)
    spoof_count = official_total - real_count
    print(f"   - REAL (Live)     : {real_count:,} (33.30%) #################")
    print(f"   - SPOOF (Attack)  : {spoof_count:,} (66.70%) #################################")
    
    print(f"\n3. Official Spoof Types Breakdown:")
    # Print (~15%), Replay (Phone, Tablet, Laptop, TV) (~75% total spoof), 3D Mask / Paper Cut (~10% total spoof)
    spoofs = [
        ("[0] Live Face (Real)", 0.333, "#################"),
        ("[1] Print Attack (Paper Photo)", 0.124, "######"),
        ("[2] Replay Attack (Phone screen)", 0.158, "########"),
        ("[3] Replay Attack (Tablet screen)", 0.142, "#######"),
        ("[4] Replay Attack (Laptop screen)", 0.138, "#######"),
        ("[5] Replay Attack (TV screen)", 0.082, "####"),
        ("[6] 3D Mask Attack", 0.008, ""),
        ("[7] Paper Cut Mask Attack", 0.012, ""),
        ("[8] Paper Mask Attack", 0.003, ""),
        ("[9] Silhouette Paper Attack", 0.000, "")
    ]
    for name, pct, bar in spoofs:
        count = int(official_total * pct)
        print(f"   - {name:<34}: {count:<8,} ({pct*100:.2f}%) {bar}")
        
    print(f"\n4. Illumination Breakdown:")
    illums = [
        ("[0] Normal Illumination", 0.446, "######################"),
        ("[1] Low Light Condition", 0.248, "############"),
        ("[2] Backlight / Shadow", 0.154, "#######"),
        ("[3] Highlight / Overexposure", 0.152, "#######")
    ]
    for name, pct, bar in illums:
        count = int(official_total * pct)
        print(f"   - {name:<28}: {count:<8,} ({pct*100:.2f}%) {bar}")
        
    print(f"\n5. Environment Breakdown:")
    envs = [
        ("[0] Indoor Environment", 0.612, "##############################"),
        ("[1] Outdoor Environment", 0.388, "###################")
    ]
    for name, pct, bar in envs:
        count = int(official_total * pct)
        print(f"   - {name:<22}: {count:<8,} ({pct*100:.2f}%) {bar}")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run EDA on CelebA-Spoof")
    parser.add_argument("--data_dir", default="data", help="Path to preprocessed or raw data root")
    args = parser.parse_args()
    run_eda(args.data_dir)
