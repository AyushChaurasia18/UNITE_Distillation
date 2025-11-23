import os
import sys
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger

# Add the parent directory to sys.path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from src.models import create_student_model, ProjectionHead, clip_loss
from src.dataset import PrecomputedPatchDataset # Ensure this class is in src/dataset.py

# ================= Distillation Lightning Module =================
class DistillationModule(pl.LightningModule):
    def __init__(self, lr=1e-4, embedding_dim=1536, proj_dim=384, temperature=0.07):
        super().__init__()
        self.save_hyperparameters()

        # Initialize student model using the factory function
        self.student = create_student_model()

        # Initialize projection head
        self.proj = ProjectionHead(in_dim=embedding_dim, out_dim=proj_dim)
        self.temperature = temperature

    def forward(self, x):
        return self.student(x)

    def training_step(self, batch, batch_idx):
        imgs, t_emb = batch['patch'], batch['embedding']
        
        # Project teacher embeddings to student dimension
        t_emb_proj = self.proj(t_emb)
        
        # Get student embeddings
        s_emb = self(imgs)
        
        # Calculate symmetric contrastive loss
        loss = clip_loss(s_emb, t_emb_proj, self.temperature)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, t_emb = batch['patch'], batch['embedding']
        
        t_emb_proj = self.proj(t_emb)
        s_emb = self(imgs)
        
        loss = clip_loss(s_emb, t_emb_proj, self.temperature)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # Optimize both student and projection head parameters
        optimizer = torch.optim.AdamW(
            list(self.student.parameters()) + list(self.proj.parameters()),
            lr=self.hparams.lr, 
            weight_decay=1e-4, 
            fused=True if torch.cuda.is_available() else False
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]


# ================= Collate Function =================
def collate_fn(batch):
    # Stack patches and embeddings into batch tensors
    patches = torch.stack([b['patch'] for b in batch])
    embeddings = torch.stack([b['embedding'] for b in batch])
    return {'patch': patches, 'embedding': embeddings}


# ================= Main Training Loop =================
if __name__ == "__main__":
    # --- Configuration ---
    OUTPUT_DIR = r"D:\Histopathology project\Processed_Patches" # Update this path if needed
    CHECKPOINT_DIR = r"D:\Histopathology project"
    MAX_EPOCHS = 50
    BATCH_SIZE = 2
    NUM_WORKERS = 6
    LR = 1e-4

    # --- Data Setup ---
    dataset = PrecomputedPatchDataset(output_dir=OUTPUT_DIR)
    total_len = len(dataset)
    val_len = int(0.1 * total_len)
    train_len = total_len - val_len

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=generator)

    print(f"📊 Data Split -> Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        collate_fn=collate_fn,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        collate_fn=collate_fn,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    # --- Logging and Callbacks ---
    tb_logger = TensorBoardLogger("lightning_logs", name="distillation_run")

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    rich_progress = RichProgressBar()

    callbacks = [checkpoint_callback, lr_monitor, rich_progress]

    # --- Trainer Setup ---
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1, # Use 1 GPU
        precision='16-mixed' if torch.cuda.is_available() else '32', # Mixed precision for speed
        log_every_n_steps=50,
        callbacks=callbacks,
        logger=tb_logger,
        gradient_clip_val=1.0 # Added gradient clipping for stability
    )

    # --- Initialize Model ---
    model = DistillationModule(lr=LR)

    # --- Resume Training Logic ---
    # Check for existing checkpoints to resume automatically
    last_ckpt = os.path.join(CHECKPOINT_DIR, "last.ckpt")
    if os.path.exists(last_ckpt):
        print(f"🔁 Resuming from last checkpoint: {last_ckpt}")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=last_ckpt)
    else:
        print("🚀 Starting training from scratch...")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save final weights manually (optional, as checkpoint callback handles this)
    final_path = os.path.join(CHECKPOINT_DIR, "distillation_final.ckpt")
    trainer.save_checkpoint(final_path)
    print(f"✅ Training complete. Final weights saved to: {final_path}")