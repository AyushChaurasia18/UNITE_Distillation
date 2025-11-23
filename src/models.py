import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, SwiGLUPacked

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.fc(x)
        x = self.norm(x)
        return x

def clip_loss(student_emb, teacher_emb, temperature=0.07):
    student_emb = F.normalize(student_emb, dim=-1)
    teacher_emb = F.normalize(teacher_emb, dim=-1)
    logits = (student_emb @ teacher_emb.t()) / temperature
    labels = torch.arange(len(logits), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2

def create_student_model():
    return VisionTransformer(
        img_size=224, patch_size=14, embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4.0, num_classes=0, class_token=False, no_embed_class=True,
        mlp_layer=SwiGLUPacked, act_layer=nn.SiLU, init_values=1e-5,
        reg_tokens=8, dynamic_img_size=True, global_pool='avg'
    )