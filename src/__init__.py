
try:
    from .models import create_student_model, ProjectionHead, clip_loss
    from .dataset import PrecomputedPatchDataset
except ImportError:
    pass

__version__ = "1.0.0"
__author__ = "Ayush Chaurasia"