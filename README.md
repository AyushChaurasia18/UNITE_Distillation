UNITE: Distilling UNI-2 into a Lightweight Vision Transformer for Histopathology

UNITE (UNified Image-to-Text Embedding Distillation) is a lightweight student Vision Transformer (ViT-Small) distilled from the large UNI-2 foundation model.

It uses a symmetric CLIP-style contrastive alignment loss to replicate the geometry of the teacher embedding space—without needing access to the teacher’s weights.

🚀 Key Highlights
Efficiency

Compresses UNI-2 (~681M parameters) → UNITE (~19M parameters)
35× reduction

Specialization

Outperforms the teacher on linear probing (BACC) for CPTAC-OV
→ Distillation acts as domain specialization

Black-Box Distillation

Requires only teacher embeddings, not weights

Works with closed-source or API-only foundation models

📂 Repository Structure
UNITE-Distillation/
├── src/
│   ├── models.py             # Student ViT + CLIP-style alignment loss
│   ├── dataset.py            # PyTorch Dataset class
│   └── utils.py              # H5 dimension fixing & helpers
├── scripts/
│   ├── 1_preprocess_wsi.py   # Patch extraction from .svs
│   ├── 2_train.py            # Distillation training loop
│   ├── 3_extract_student.py  # Student inference
│   ├── 4_extract_uni2.py     # Teacher (UNI-2) baseline extraction
│   ├── 5_pool_features.py    # Patch → slide/case pooling
│   └── 6_benchmark.py        # PathoBench: retrieval + linear probe
├── requirements.txt
└── README.md

🛠️ Installation & Setup
1. Clone the repository
git clone https://github.com/AyushChaurasia18/UNITE_Distillation.git
cd UNITE_Distillation

2. Install dependencies
pip install -r requirements.txt

3. Install external frameworks

This project uses TRIDENT for WSI processing and PathoBench for evaluation.

pip install git+https://github.com/mahmoodlab/TRIDENT.git
pip install git+https://github.com/mahmoodlab/Patho-Bench.git

🏃‍♂️ Usage Pipeline
Step 1: Data Pre-processing

Extract patches from .svs Whole Slide Images.

# Edit scripts/1_preprocess_wsi.py to set SVS_DIR
python scripts/1_preprocess_wsi.py

Step 2: Distillation Training

Train UNITE using pre-computed teacher embeddings.

# Logs saved to lightning_logs/
python scripts/2_train.py

Step 3: Feature Extraction

Generate student embeddings for the full dataset.

# Edit script to use your trained .ckpt
python scripts/3_extract_student.py

Step 4: Benchmarking

Evaluate retrieval & linear probing performance.

python scripts/5_pool_features.py
python scripts/6_benchmark.py

📊 Results (CPTAC-OV)
Model	Params	Embedding Dim	Retrieval (mAP@1)	Linear Probe (BACC)
UNI-2 (Teacher)	~681M	1536	0.400	0.346
UNITE (Student)	~19M	384	0.352	0.412

UNITE achieves 35× compression and outperforms the teacher in linear separability.

💻 Computational Requirements
Training

NVIDIA H100 80GB recommended

Batch size 64+

bf16 mixed precision

Total time: ~23 hours for 150 epochs

Inference

Works on consumer GPUs (RTX 3090/4090)

🙌 Acknowledgements

TRIDENT & PathoBench – Mahmood Lab, Harvard Medical School

Mentor: Prof. Maitrik Shah (Ahmedabad University)
