import torch
import torch.nn.functional as F


def smoothness_loss(disp, img):
    """计算边缘感知的平滑度损失
    
    鼓励视差图在图像边缘处不连续，在平滑区域保持平滑。
    通过图像梯度调制视差梯度损失，使视差边界与图像边界对齐。
    
    参数:
        disp (Tensor): 预测的视差图，形状为 [B, 1, H, W]
        img (Tensor): 输入图像，形状为 [B, C, H, W]
        
    返回:
        Tensor: 边缘感知的平滑度损失
    """
    # 确保输入图像已归一化到0-1范围
    if img.max() > 1.0:
        print("Warning: Image may not be normalized. Expected range: [0,1]")

    # 视差归一化（消除绝对尺度的影响，使损失关注局部结构)
    mean_disp = disp.mean(2, True).mean(3, True)
    norm_disp = disp / (mean_disp + 1e-7)
    
    # 计算视差梯度
    disp_dx = torch.abs(norm_disp[:, :, :, :-1] - norm_disp[:, :, :, 1:])
    disp_dy = torch.abs(norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :])
    
    # 计算图像梯度 (对归一化后的图像)
    img_dx = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    img_dy = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
    
    # 边缘感知权重（图像梯度处权重较小）
    weights_x = torch.exp(-img_dx)
    weights_y = torch.exp(-img_dy)
    
    # 应用权重
    smoothness_x = disp_dx * weights_x
    smoothness_y = disp_dy * weights_y

    loss = torch.mean(smoothness_x) + torch.mean(smoothness_y)
    
    return loss