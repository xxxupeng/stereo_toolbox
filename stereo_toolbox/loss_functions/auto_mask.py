import torch
import torch.nn.functional as F

from .photometric_loss import photometric_loss


def auto_mask(left_image, right_image, disp, denorm=False):
    if denorm:
        mean = torch.tensor([0.485, 0.456, 0.406], device=left_image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=left_image.device).view(1, 3, 1, 1)
        left_image = left_image * std + mean
        right_image = right_image * std + mean

    reproj_error = photometric_loss(left_image, right_image, disp.detach(), enable_mask=False)
    identity_error = photometric_loss(left_image, right_image, enable_mask=False)

    return reproj_error < identity_error