import torch
import torch.nn.functional as F

def warp_right_to_left(right_image, disp):
    """Warp the right image to the left image's viewpoint using the disparity map.
    
    This function warps the right image to the left image's viewpoint based on the given disparity map, 
    enabling view synthesis. It is a fundamental operation for computing reconstruction loss in self-supervised stereo matching.
    
    Args:
        right_image (Tensor): The right image with shape [B, C, H, W].
        disp (Tensor): The predicted disparity map with shape [B, 1, H, W].
        
    Returns:
        Tensor: The right image warped to the left viewpoint.
    """
    batch_size, _, height, width = right_image.size()
    
    # Generate grid coordinates
    device = disp.device
    x_base = torch.linspace(0, 1, width, device=device).repeat(batch_size, height, 1)
    y_base = torch.linspace(0, 1, height, device=device).repeat(batch_size, width, 1).transpose(1, 2)
    flow_field = torch.stack((x_base - disp.squeeze(1) / (width-1), y_base), dim=3)
    
    # Use grid_sample for reprojection
    warped_right = F.grid_sample(right_image, 
                                 (flow_field * 2 - 1),
                                 mode='bilinear', 
                                 padding_mode='zeros')
    
    return warped_right


def ssim(x, y, window_size=7, pad_mode='reflect'):
    """Compute Structural Similarity (SSIM) loss.
    
    SSIM considers structural information in images and is more robust to illumination changes compared to simple L1 loss.
    The returned value is the SSIM distance loss: (1-SSIM)/2, ranging from [0,1], where smaller values indicate higher similarity.
    
    Args:
        x (Tensor): The first image with shape [B, C, H, W].
        y (Tensor): The second image with shape [B, C, H, W].
        window_size (int): The window size for SSIM computation.
        
    Returns:
        Tensor: SSIM distance loss with shape [B, C, H, W].
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # reflection padding
    pad = window_size // 2
    x_padded = F.pad(x, (pad, pad, pad, pad), mode=pad_mode)
    y_padded = F.pad(y, (pad, pad, pad, pad), mode=pad_mode)
    
    mu_x = F.avg_pool2d(x_padded, window_size, stride=1)
    mu_y = F.avg_pool2d(y_padded, window_size, stride=1)
    
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = F.avg_pool2d(x_padded * x_padded, window_size, stride=1) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(y_padded * y_padded, window_size, stride=1) - mu_y_sq
    sigma_xy = F.avg_pool2d(x_padded * y_padded, window_size, stride=1) - mu_xy
    
    ssim_n = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    
    return torch.clamp((1 - ssim_n / ssim_d) / 2, 0, 1)



def auto_mask_loss(loss, left_image, right_image, disp):
    """Automatic masking loss function for handling challenging regions in self-supervised stereo matching.
    
    This function implements an automatic masking mechanism by comparing the reprojection error 
    based on predicted disparity with the direct image matching error. It identifies and excludes 
    the following problematic regions:
    1. Distant/infinity regions (e.g., sky): Disparity approaches zero, making accurate estimation difficult.
    2. Low-texture regions: Multiple disparity values may produce similar reprojection results.
    3. Non-overlapping regions: Areas visible in the left image but not present in the right image.
    
    Principle: When the reprojection error from predicted disparity is not smaller than the error 
    from directly using the untransformed right image, the region is likely one of the problematic 
    areas mentioned above. These regions should be excluded during training to avoid introducing noise.
    
    Args:
        loss (Tensor): Original pixel-wise loss with shape [B, 1, H, W].
        left_image (Tensor): Left image with shape [B, C, H, W].
        right_image (Tensor): Right image with shape [B, C, H, W].
        disp (Tensor): Predicted disparity map with shape [B, 1, H, W].
        
    Returns:
        Tensor: Loss after applying automatic masking.
    """
    reproj_error = photometric_loss(left_image, right_image, disp.detach())
    identity_error = photometric_loss(left_image, right_image)
    
    mask = (reproj_error < identity_error).float()
    valid_pixels = torch.sum(mask) + 1e-8
    
    loss = torch.sum(loss * mask) / valid_pixels
    
    return loss


def photometric_loss(left_image, right_image, disp=None, ssim_weight=0.85, auto_mask=False):
    """Compute photometric consistency loss.
    
    Combines SSIM loss and L1 loss as a weighted sum to evaluate the accuracy of disparity prediction.
    SSIM loss captures structural information, while L1 loss captures detailed information.
    
    Args:
        left_image (Tensor): The left image with shape [B, C, H, W].
        right_image (Tensor): The right image with shape [B, C, H, W].
        disp (Tensor): The predicted disparity map with shape [B, 1, H, W].
        ssim_weight (float): The weight for SSIM loss, default is 0.85.
        
    Returns:
        Tensor: The photometric consistency loss with shape [B, C, H, W].
    """
    if disp is None:
        warped_right_image = right_image
    else:
        warped_right_image = warp_right_to_left(right_image, disp)

    loss = ssim_weight * ssim(left_image, warped_right_image).mean(1, True) + (1-ssim_weight) * torch.abs(left_image - warped_right_image).mean(1, True)

    if auto_mask:
        loss = auto_mask_loss(loss, left_image, right_image, disp)
        return loss
    else:
        return loss.mean() 

