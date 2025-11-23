import os
import glob
import shutil
import gc
import warnings
import h5py
import numpy as np
import pandas as pd
import openslide
import torchvision.transforms as transforms
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Global Configuration ---
# Define transforms globally so they are pickleable by parallel workers
PATCH_PRE_SAVE_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
])

# ==== 2️⃣ Utility: Extract Patches for ONE Slide (Robust Error Handling) ====
def extract_patches_from_slide(wsi_path, h5_path, output_dir, patch_size=512, level=0):
    slide_id = os.path.basename(wsi_path).replace(".svs", "")
    
    # --- 1. CRITICAL ERROR TRAPPING FOR OPENSIDE (SVS Open) ---
    try:
        slide = openslide.OpenSlide(wsi_path)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: Failed to open SVS file {slide_id}. Skipping. Error: {e}", flush=True)
        return []

    # --- 2. CRITICAL ERROR TRAPPING FOR H5PY (H5 Read) ---
    try:
        with h5py.File(h5_path, "r") as h5f:
            coords = np.array(h5f["coords"]).squeeze()
            embeddings = np.array(h5f["features"]).squeeze()
    except Exception as e:
        slide.close()
        print(f"\n❌ CRITICAL ERROR: Failed to read H5 file for {slide_id}. Skipping. Error: {e}", flush=True)
        return []
    
    # Safety check for empty data
    if coords.ndim == 1 and coords.shape[0] == 0:
        slide.close()
        print(f"\n   ⚠️ Skipping {slide_id}: No coordinates found in .h5 file.", flush=True)
        return []
    
    # Assert check
    if coords.ndim > 1:
        if coords.shape[0] != embeddings.shape[0]:
             raise AssertionError(f"Coord/Embedding length mismatch in {slide_id}")
    elif coords.ndim == 0: # Handle edge case of single coord squeezed to scalar
         if embeddings.ndim == 0 and coords.size == embeddings.size:
             pass
         else:
             raise AssertionError(f"Coord/Embedding dimension mismatch in {slide_id} after squeeze")

    # Output directory setup
    img_dir = os.path.join(output_dir, slide_id, "images")
    emb_dir = os.path.join(output_dir, slide_id, "embeddings")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)

    records = []

    # Ensure coords is iterable
    if coords.ndim == 1:
        coords = coords[np.newaxis, :]
        embeddings = embeddings[np.newaxis, :]

    # --- 3. ERROR TRAPPING FOR INDIVIDUAL PATCHES ---
    for i in range(len(coords)):      
        x, y = coords[i]
        patch_id = f"{slide_id}_patch_{i:06d}"

        try:
            # 1. Read patch from WSI
            patch = slide.read_region((int(x), int(y)), level, (patch_size, patch_size)).convert("RGB")

            # 2. Apply pre-save transforms
            transformed_patch = PATCH_PRE_SAVE_TRANSFORM(patch)

            # 3. Save transformed patch as compressed JPEG
            img_path = os.path.join(img_dir, f"{patch_id}.jpg")
            transformed_patch.save(img_path, quality=90, optimize=True)

            # 4. Save corresponding embedding
            emb_path = os.path.join(emb_dir, f"{patch_id}.npy")
            np.save(emb_path, embeddings[i].astype(np.float32))

            # 5. Store metadata
            records.append({
                "patch_id": patch_id,
                "slide_id": slide_id,
                "x": int(x),
                "y": int(y),
                "image_path": img_path,
                "embedding_path": emb_path
            })
        
        except Exception as e:
            print(f"\n   ⚠️ Patch read/save error for {patch_id} (coords: {x}, {y}). Error: {e}", flush=True)
            
    slide.close()
    del coords, embeddings, slide 
    gc.collect() 
    return records


# ==== 3️⃣ Scan and Check Function ====
def scan_and_check_folders(svs_dir, h5_dir, output_dir):
    slides_to_process = []
    if os.path.exists(output_dir):
        processed_slide_ids = set(os.listdir(output_dir))
    else:
        processed_slide_ids = set()
        
    reprocessed_count = 0
    skipped_h5_count = 0
    
    # Iterate over the SVS directory to find all WSI files
    for file in os.listdir(svs_dir):
        if file.endswith(".svs"):
            slide_id = file.replace(".svs", "")
            wsi_path = os.path.join(svs_dir, file)
            h5_path = os.path.join(h5_dir, slide_id + ".h5")
            output_folder_path = os.path.join(output_dir, slide_id)

            # Check 1: Does the H5 file exist?
            if not os.path.exists(h5_path):
                skipped_h5_count += 1
                continue

            # Check 2: Has it been completely processed before?
            if slide_id in processed_slide_ids and os.path.isdir(output_folder_path):
                try:
                    # Get expected count from H5
                    with h5py.File(h5_path, "r") as h5f:
                        coords_shape = np.array(h5f["coords"]).squeeze().shape
                        expected_count = coords_shape[0] if coords_shape and coords_shape[0] > 0 else 1
                    
                    # Get actual count of saved patches
                    actual_count = len(glob.glob(os.path.join(output_folder_path, "images", "*.jpg")))

                    if actual_count < expected_count:
                        # Incomplete: DELETE AND REPROCESS
                        shutil.rmtree(output_folder_path)
                        print(f"   🗑️ Incomplete slide: {slide_id} ({actual_count}/{expected_count}). Reprocessing.")
                        reprocessed_count += 1
                        slides_to_process.append((wsi_path, h5_path))
                        continue 
                    continue # Complete
                
                except Exception as e:
                    print(f"   ⚠️ Integrity check failed for {slide_id}. Reprocessing. Error: {e}")
                    slides_to_process.append((wsi_path, h5_path))
                    continue

            # Check 3: New slide to process
            slides_to_process.append((wsi_path, h5_path))

    return slides_to_process, reprocessed_count, skipped_h5_count


if __name__ == "__main__":
    # --- Configuration ---
    # NOTE: You can change these paths here or use argparse in the future
    NUM_WORKERS = 6 
    SVS_DIR = r"D:\Histopathology project\OV"
    H5_DIR = r"D:\Histopathology project\OV_h5"
    OUTPUT_DIR = r'D:\Histopathology project\Processed_Patches' 
    PATCH_SIZE = 512
    LEVEL = 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Run the scan and check ---
    print("Scanning folders...")
    slides_to_process, reprocessed_count, skipped_h5_count = scan_and_check_folders(SVS_DIR, H5_DIR, OUTPUT_DIR)
    new_slides_count = len(slides_to_process) - reprocessed_count

    print(f"🧩 Found {new_slides_count} new slides to process.")
    print(f"   ({reprocessed_count} incomplete slides will be reprocessed.)")
    print(f"   ({skipped_h5_count} SVS files skipped because their H5 was missing.)\n")

    # ==== 4️⃣ Process all slides (Parallelized) ====
    all_records = []
    if slides_to_process:
        print(f"Starting parallel processing with {NUM_WORKERS} workers...")

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)

            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {executor.submit(
                    extract_patches_from_slide, wsi_path, h5_path, OUTPUT_DIR, patch_size=PATCH_SIZE, level=LEVEL
                ): wsi_path for wsi_path, h5_path in slides_to_process}

                for future in tqdm(as_completed(futures), total=len(slides_to_process), desc="Total Slides Processed"):
                    try:
                        records = future.result()
                        all_records.extend(records)
                    except Exception as e:
                        failed_slide = futures[future]
                        print(f"\n❌ Error processing slide {os.path.basename(failed_slide)}: {e}")

        # ==== 5️⃣ Save global CSV ====
        index_csv = os.path.join(OUTPUT_DIR, "patch_index.csv")
        all_records_df = pd.DataFrame(all_records)

        if os.path.exists(index_csv):
            try:
                existing_df = pd.read_csv(index_csv)
            except pd.errors.EmptyDataError:
                print("⚠️ Warning: Existing patch_index.csv is empty. Starting fresh index.")
                existing_df = pd.DataFrame() 
            
            if not existing_df.empty:
                processed_slide_ids_in_run = {os.path.basename(wsi_path).replace(".svs", "") for wsi_path, _ in slides_to_process}
                filtered_df = existing_df[~existing_df['slide_id'].isin(processed_slide_ids_in_run)]
            else:
                filtered_df = existing_df
            
            combined_df = pd.concat([filtered_df, all_records_df], ignore_index=True)
        else:
            combined_df = all_records_df

        combined_df.to_csv(index_csv, index=False)

        print(f"\n✅ Done! Extracted {len(all_records)} new patches.")
        print(f"   Total patches in index: {len(combined_df)}")
        print(f"📁 Patches stored under: {OUTPUT_DIR}")
        print(f"📄 Global index saved at: {index_csv}")
    else:
        print("✅ No new slides to process.")