import torch
import torch.nn.functional as F


def warp_right_to_left(right_image, disp):
    """将右图像根据视差图重投影到左图像视角
    
    使用给定的视差图将右图像变形到左图像的视角，实现视图合成。
    这是自监督立体匹配中计算重建损失的基础操作。
    
    参数:
        right_image (Tensor): 右图像，形状为 [B, C, H, W]
        disp (Tensor): 预测的视差图，形状为 [B, 1, H, W]
        
    返回:
        Tensor: 重投影到左视角的右图像
    """
    batch_size, _, height, width = right_image.size()
    
    # 生成网格坐标
    device = disp.device
    x_base = torch.linspace(0, 1, width, device=device).repeat(batch_size, height, 1)
    y_base = torch.linspace(0, 1, height, device=device).repeat(batch_size, width, 1).transpose(1, 2)
    flow_field = torch.stack((x_base - disp.squeeze(1) / (width-1), y_base), dim=3)
    
    # 使用grid_sample进行重投影
    warped_right = F.grid_sample(right_image, 
                                 (flow_field * 2 - 1),  # 转换到[-1,1]范围
                                 mode='bilinear', 
                                 padding_mode='zeros')
    
    return warped_right


def ssim(x, y, window_size=7, pad_mode='reflect'):
    """计算结构相似性(SSIM)损失
    
    SSIM考虑图像结构信息，比简单L1损失对光照变化更鲁棒。
    返回的是SSIM距离损失：(1-SSIM)/2，范围为[0,1]，值越小表示图像越相似。
    
    参数:
        x (Tensor): 第一个图像，形状为 [B, C, H, W]
        y (Tensor): 第二个图像，形状为 [B, C, H, W]
        window_size (int): SSIM计算的窗口大小
        
    返回:
        Tensor: SSIM距离损失，形状为 [B, C, H, W]
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 使用反射填充
    pad = window_size // 2
    x_padded = F.pad(x, (pad, pad, pad, pad), mode=pad_mode)
    y_padded = F.pad(y, (pad, pad, pad, pad), mode=pad_mode)
    
    # 无填充的平均池化（因为已经填充）
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


def photometric_loss(left_image, right_image, disp=None, ssim_weight=0.85):
    """计算光度一致性损失
    
    结合SSIM损失和L1损失的加权和，用于评估视差预测的准确性。
    SSIM损失捕获结构信息，L1损失捕获细节信息。
    
    参数:
        left_image (Tensor): 左图像，形状为 [B, C, H, W]
        right_image (Tensor): 右图像，形状为 [B, C, H, W]
        disp (Tensor): 预测的视差图，形状为 [B, 1, H, W]
        ssim_weight (float): SSIM损失的权重，默认为0.85
        
    返回:
        Tensor: 光度一致性损失，形状为 [B, 1, H, W]
    """
    if disp is None:
        warped_right_image = right_image
    else:
        warped_right_image = warp_right_to_left(right_image, disp)

    # 计算光度一致性损失
    return ssim_weight * ssim(left_image, warped_right_image).mean(1, True) + (1-ssim_weight) * torch.abs(left_image - warped_right_image).mean(1, True)
