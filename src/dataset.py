import torch
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class PrecomputedPatchDataset(Dataset):
    """Dataset returning patch image + precomputed embedding."""
    def __init__(self, output_dir, transform=None):
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])
        self.samples = []

        slide_dirs = [
            os.path.join(output_dir, d)
            for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d))
        ]

        for slide_dir in tqdm(slide_dirs, desc="📂 Scanning slides"):
            img_dir = os.path.join(slide_dir, "images")
            emb_dir = os.path.join(slide_dir, "embeddings")

            if not os.path.exists(img_dir) or not os.path.exists(emb_dir):
                continue

            patch_ids = sorted([
                f.replace(".jpg", "")
                for f in os.listdir(img_dir)
                if f.endswith(".jpg")
            ])

            for pid in patch_ids:
                img_path = os.path.join(img_dir, pid + ".jpg")
                emb_path = os.path.join(emb_dir, pid + ".npy")
                if os.path.exists(img_path) and os.path.exists(emb_path):
                    self.samples.append({
                        "patch_id": pid,
                        "image_path": img_path,
                        "embedding_path": emb_path
                    })

        print(f"✅ Finished scanning {len(self.samples)} patches.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        patch = Image.open(sample["image_path"]).convert("RGB")
        patch = self.transform(patch)

        embedding = np.load(sample["embedding_path"]).astype(np.float32)
        embedding = torch.tensor(embedding, dtype=torch.float32)

        return {"patch": patch, "embedding": embedding, "patch_id": sample["patch_id"]}

    def get_transforms():
        return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])