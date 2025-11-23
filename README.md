UNITE: Distilling UNI-2 into a Lightweight Vision Transformer for Histopathology

UNITE (UNified Image-to-Text Embedding distillation) is a lightweight student Vision Transformer (ViT-Small) distilled from the massive UNI-2 foundation model.

Using a symmetric CLIP-style contrastive alignment loss, UNITE learns to approximate the geometry of the teacher's embedding space without requiring access to the teacher's internal weights.

🚀 Key Highlights

Efficiency: Compresses the ~681M parameter UNI-2 teacher into a ~19M parameter student (35x reduction).

Specialization: The student model outperforms the teacher on linear probing (BACC) for the target cohort (CPTAC-OV), demonstrating that distillation can act as a form of domain specialization.

Black-Box Distillation: Training requires only the teacher's output embeddings, making it compatible with API-based or closed-source foundation models.

📂 Repository Structure

UNITE-Distillation/
├── src/                  # Source code for models and data loading
│   ├── models.py         # Student ViT architecture and CLIP Loss
│   ├── dataset.py        # PyTorch Dataset class
│   └── utils.py          # Helper functions (H5 dimension fixing)
├── scripts/              # Executable scripts for the full pipeline
│   ├── 1_preprocess_wsi.py  # Patch extraction from .svs slides
│   ├── 2_train.py           # Main distillation training loop (Lightning)
│   ├── 3_extract_student.py # Inference with trained student
│   ├── 4_extract_uni2.py    # Baseline feature extraction
│   ├── 5_pool_features.py   # Aggregate patches to slide/case level
│   └── 6_benchmark.py       # Run Linear Probe & Retrieval (PathoBench)
├── requirements.txt      # Python dependencies
└── README.md             # This file


🛠️ Installation & Setup

Clone the repository:

git clone [https://github.com/AyushChaurasia18/UNITE_Distillation.git](https://github.com/AyushChaurasia18/UNITE_Distillation.git)
cd UNITE_Distillation


Install dependencies:

pip install -r requirements.txt


Install External Frameworks:
This project relies on TRIDENT for WSI processing and PathoBench for evaluation.

pip install git+[https://github.com/mahmoodlab/TRIDENT.git](https://github.com/mahmoodlab/TRIDENT.git)
pip install git+[https://github.com/mahmoodlab/Patho-Bench.git](https://github.com/mahmoodlab/Patho-Bench.git)


🏃‍♂️ Usage Pipeline

Step 1: Data Pre-processing

Extract raw patches from Whole Slide Images (.svs) to prepare for training.

# Edit scripts/1_preprocess_wsi.py to set your SVS_DIR
python scripts/1_preprocess_wsi.py


Step 2: Distillation Training

Train the UNITE student model to mimic pre-computed teacher embeddings.

# Logs will be saved to lightning_logs/
python scripts/2_train.py


Step 3: Feature Extraction

Generate embeddings for the entire cohort using the trained student model.

# Edit script to point to your new .ckpt file
python scripts/3_extract_student.py


Step 4: Benchmarking

Run standardized evaluation tasks (Retrieval and Linear Probing) to compare Student vs. Teacher.

# Pools features and runs PathoBench
python scripts/5_pool_features.py
python scripts/6_benchmark.py


📊 Results (CPTAC-OV Cohort)

Model

Parameters

Embedding Dim

Retrieval (mAP@1)

Linear Probe (BACC)

UNI-2 (Teacher)

~681M

1536

0.400

0.346

UNITE (Student)

~19M

384

0.352

0.412

The student model achieves 35x compression while surpassing the teacher in linear separability for the specific task.

💻 Computational Requirements

Training: Single NVIDIA H100 (80GB) recommended for batch size 64+ with bf16-mixed precision. Training takes approx. 23 hours for 150 epochs.

Inference: Can be run on consumer GPUs (e.g., RTX 3090/4090).

🙌 Acknowledgements

TRIDENT & PathoBench: Tools developed by the Mahmood Lab at Harvard Medical School.

Project Mentor: Prof. Maitrik Shah (Ahmedabad University).