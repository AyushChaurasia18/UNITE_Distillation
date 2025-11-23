import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Add the parent directory to sys.path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your model definition to ensure architecture matches training
from src.models import create_student_model

# TRIDENT Imports
try:
    from trident.patch_encoder_models import CustomInferenceEncoder
    from trident.Processor import Processor
    from trident.segmentation_models import segmentation_model_factory
except ImportError:
    raise ImportError("TRIDENT is not installed. Please clone and install it first.")

# ================= Configuration =================
WSI_DIR = r"D:\Histopathology project\OV"               # Input SVS folder
OUTPUT_DIR = r"D:\Histopathology project\WSI_patch_features" # Output folder
STUDENT_WEIGHTS_PATH = r"C:\Users\Hp\Downloads\student_model.pth" # Path to your .pth or .ckpt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TRIDENT Settings
PATCH_SIZE = 256
MAGNIFICATION = 20
BATCH_LIMIT = 8  # Lower this if you run out of GPU memory

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Load Student Model ---
    print(f"Loading student model architecture...")
    student_model = create_student_model()
    
    print(f"Loading weights from: {STUDENT_WEIGHTS_PATH}")
    # Handle both full checkpoint (.ckpt) and state dictionary (.pth)
    state_dict = torch.load(STUDENT_WEIGHTS_PATH, map_location='cpu')
    if 'state_dict' in state_dict:
        # If loading from Lightning checkpoint, strip the "student." prefix if necessary
        state_dict = {k.replace("student.", ""): v for k, v in state_dict['state_dict'].items() if "student." in k}
    
    student_model.load_state_dict(state_dict)
    student_model.to(DEVICE)
    student_model.eval()
    print("✅ Student model loaded successfully.")

    # --- 2. Define Transforms ---
    # Must match the normalization used during training (ImageNet stats)
    student_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    # --- 3. Create TRIDENT Custom Encoder ---
    custom_patch_encoder = CustomInferenceEncoder(
        enc_name='MyStudentViT',
        model=student_model,
        transforms=student_transforms,
        precision=torch.float16 # Use float16 for faster inference
    )

    # --- 4. Initialize Processor ---
    processor = Processor(
        job_dir=OUTPUT_DIR,
        wsi_source=WSI_DIR,
        reader_type='openslide'
    )

    # --- 5. Run Pipeline ---
    
    # Step A: Segmentation
    print("\n--- Starting Stage 1: Segmentation (HEST) ---")
    segmentation_model = segmentation_model_factory('hest')
    processor.run_segmentation_job(
        segmentation_model,
        device=DEVICE
    )

    # Step B: Patching
    print(f"\n--- Starting Stage 2: Patching ({PATCH_SIZE}px at {MAGNIFICATION}x) ---")
    processor.run_patching_job(
        target_magnification=MAGNIFICATION,
        patch_size=PATCH_SIZE,
        overlap=0
    )

    # Step C: Feature Extraction
    print("\n--- Starting Stage 3: Feature Extraction ---")
    coords_dir_name = f"{MAGNIFICATION}x_{PATCH_SIZE}px_0px_overlap"
    
    processor.run_patch_feature_extraction_job(
        coords_dir=coords_dir_name,
        patch_encoder=custom_patch_encoder,
        device=DEVICE,
        saveas='h5',
        batch_limit=BATCH_LIMIT
    )

    print(f"\n✅✅✅ Student Extraction Complete! ✅✅✅")
    print(f"Embeddings saved in: {os.path.join(OUTPUT_DIR, coords_dir_name, 'MyStudentViT')}")

if __name__ == "__main__":
    main()