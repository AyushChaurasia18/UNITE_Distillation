import os
import sys
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import fix_h5_shape

# Try imports for PathoBench (ensure it's installed)
try:
    from patho_bench.SplitFactory import SplitFactory
except ImportError:
    print("⚠️ PathoBench not found. Please install it via: pip install git+https://github.com/mahmoodlab/Patho-Bench.git")
    sys.exit(1)

# ================= Configuration =================
# You can switch this to 'MyStudentViT' when processing student features
MODEL_NAME = "uni_v2" 
DATASET_NAME = "cptac_ov"
TASK_NAME = "Immune_class"

# Input/Output Paths (Adjust these to match your folder structure)
BASE_FEATURE_DIR = r"D:\Histopathology project\uni2_extracted_features" 
OUTPUT_POOL_DIR = r"D:\Histopathology project\Pooled_Features"

LEVELS_TO_POOL = [
    "20x_256px_0px_overlap",
    "40x_512px_0px_overlap",
    "20x_512px_0px_overlap"
]

def main():
    # 1. Load Split Data to get Case ID -> Slide ID mapping
    print("Loading split data...")
    try:
        path_to_split, _ = SplitFactory.from_hf('./benchmark_splits', DATASET_NAME, TASK_NAME)
        split_df = pd.read_csv(path_to_split, sep='\t')
        case_to_slides_map = split_df.groupby('case_id')['slide_id'].apply(list)
        print(f"✅ Found {len(case_to_slides_map)} unique cases.")
    except Exception as e:
        print(f"❌ Error loading splits: {e}")
        return

    # 2. Loop through each configuration level
    for level in LEVELS_TO_POOL:
        print(f"\n{'='*40}\nPooling Level: {level}\n{'='*40}")

        # construct paths
        # Note: Adjust this logic if your student/teacher folders have slightly different naming conventions
        source_dir = os.path.join(BASE_FEATURE_DIR, level, f"features_{MODEL_NAME}")
        target_dir = os.path.join(OUTPUT_POOL_DIR, level, MODEL_NAME)
        
        os.makedirs(target_dir, exist_ok=True)

        # 3. Pool each case
        for case_id, slide_list in tqdm(case_to_slides_map.items(), desc=f"Pooling {MODEL_NAME}"):
            pooled_h5_path = os.path.join(target_dir, f"{case_id}.h5")
            
            # Skip if already exists
            if os.path.exists(pooled_h5_path):
                continue

            sum_features = None
            total_patches = 0

            # Aggregate patches from all slides for this patient
            for slide_id in slide_list:
                slide_h5 = os.path.join(source_dir, f"{slide_id}.h5")
                
                if not os.path.exists(slide_h5):
                    continue
                
                try:
                    with h5py.File(slide_h5, 'r') as f:
                        # Handle both (N, D) and (D,) shapes safely
                        feats = f['features'][:]
                        if feats.ndim == 1: 
                            feats = feats.reshape(1, -1)
                        
                        if feats.size == 0: continue

                        # Sum features for this slide
                        current_sum = feats.sum(axis=0)
                        
                        if sum_features is None:
                            sum_features = current_sum
                        else:
                            sum_features += current_sum
                            
                        total_patches += feats.shape[0]
                except Exception as e:
                    print(f"  ⚠️ Error reading {slide_id}: {e}")

            # Save pooled result
            if total_patches > 0 and sum_features is not None:
                # Calculate Mean
                mean_feat = (sum_features / total_patches).astype(np.float32)
                
                # Ensure shape is (1, D) for PathoBench compatibility
                if mean_feat.ndim == 1:
                    mean_feat = mean_feat.reshape(1, -1)

                with h5py.File(pooled_h5_path, 'w') as f:
                    f.create_dataset('features', data=mean_feat)
            else:
                # print(f"  [Warn] No features found for case {case_id}")
                pass

    print("\n✅ Pooling Complete!")

if __name__ == "__main__":
    main()