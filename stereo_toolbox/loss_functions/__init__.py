import torch
import torch.nn.functional as F

from .photometric_loss import photometric_loss
from .auto_mask import auto_mask