import os
import sys
import torch

# TRIDENT Imports
try:
    from trident.Processor import Processor
    from trident.segmentation_models import segmentation_model_factory
    from trident.patch_encoder_models import encoder_factory
except ImportError:
    raise ImportError("TRIDENT is not installed. Please clone and install it first.")

# ================= Configuration =================
WSI_DIR = r"D:\Histopathology project\OV"
OUTPUT_DIR = r"D:\Histopathology project\uni2_extracted_features"
MODEL_NAME = "uni_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define the configurations you want to benchmark
EXTRACTION_CONFIGS = [
    {"mag": 20, "size": 256, "overlap": 0}, # Standard
    {"mag": 20, "size": 512, "overlap": 0}, # Larger Context
    {"mag": 40, "size": 512, "overlap": 0}, # High Res
]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")

    # --- 1. Create Processor ---
    processor = Processor(
        job_dir=OUTPUT_DIR,
        wsi_source=WSI_DIR,
        reader_type='openslide'
    )

    # --- 2. Load UNI-2 Model ---
    print(f"\n--- Loading Patch Encoder Model: {MODEL_NAME} ---")
    try:
        # We load the object here to pass it explicitly to the processor
        patch_encoder_obj = encoder_factory(MODEL_NAME)
        print(f"✅ Patch encoder model '{MODEL_NAME}' loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading encoder model {MODEL_NAME}: {e}")
        sys.exit(1)

    # --- 3. Run Segmentation (Once for all configs) ---
    print("\n--- Starting Stage 1: Segmentation (HEST) ---")
    segmentation_model = segmentation_model_factory('hest')  
    processor.run_segmentation_job(
        segmentation_model,
        device=DEVICE
    )
    print("✅ Segmentation complete.")

    # --- 4. Loop Through Configurations ---
    for config in EXTRACTION_CONFIGS:
        target_mag = config['mag']
        patch_size = config['size']
        overlap = config['overlap']
        
        coords_dir = f"{target_mag}x_{patch_size}px_{overlap}px_overlap"
        print(f"\n{'='*40}")
        print(f"Running Configuration: {coords_dir}")
        print(f"{'='*40}")

        # Stage 2: Patching
        print("  [Step 2] Running Patching...")
        processor.run_patching_job(
            target_magnification=target_mag,
            patch_size=patch_size,
            overlap=overlap
        )

        # Stage 3: Feature Extraction
        print("  [Step 3] Running Feature Extraction...")
        processor.run_patch_feature_extraction_job(
            coords_dir=coords_dir,
            patch_encoder=patch_encoder_obj, # Pass the loaded object
            device=DEVICE,
            saveas='h5',
            batch_limit=128 if patch_size == 256 else 64 # Adjust batch size for memory
        )
        
        print(f"  ✅ Config complete. Embeddings at: {os.path.join(OUTPUT_DIR, coords_dir, 'features_' + MODEL_NAME)}")

    print(f"\n🎉🎉🎉 ALL UNI-2 EXTRACTION PIPELINES COMPLETE! 🎉🎉🎉")

if __name__ == "__main__":
    main()