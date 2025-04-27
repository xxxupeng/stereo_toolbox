import sys
sys.path.insert(0, '/home/xp/stereo_toolbox/')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import cv2
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from pathlib import Path
import tensorboard

from stereo_toolbox.datasets import *
from stereo_toolbox.models import *
from stereo_toolbox.visualization import *


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='DDP 训练示例')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮次')
    parser.add_argument('--output-dir', type=str, default='./checkpoint/', help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--sync-bn', action='store_true', help='使用同步批量归一化')
    parser.add_argument('--amp', action='store_true', help='使用混合精度训练')
    parser.add_argument('--save-every', type=int, default=1, help='每 x epochs保存一次')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--clip-grad', type=float, default=0, help='梯度裁剪阈值, 0 为不裁剪')
    parser.add_argument('--find-unused_parameters', action='store_true', help='查找未使用的参数')
    parser.add_argument('--max-disp', type=int, default=192, help='最大视差')

    return parser.parse_args()


def train_iteration(model, data, optimizer, scheduler, accelerator, args):
    optimizer.zero_grad()

    # 获取数据
    left = data['left'].to(accelerator.device)
    right = data['right'].to(accelerator.device)
    gt_disp = data['gt_disp'].to(accelerator.device).squeeze()

    mask = (gt_disp > 0) * (gt_disp < args.max_disp - 1)

    with accelerator.autocast():
        init_disp, disp_preds = model(left, right)

        n_predictions = len(disp_preds)
        loss = F.smooth_l1_loss(init_disp.squeeze()[mask], gt_disp[mask], reduction='mean')
        for i in range(n_predictions):
            loss += F.smooth_l1_loss(disp_preds[i].squeeze()[mask], gt_disp[mask], reduction='mean') / (2**i)

    accelerator.backward(loss)
    if args.clip_grad:
        accelerator.clip_grad_norm_(model.parameters(), args.clip_grad)

    optimizer.step()
    scheduler.step()

    loss = accelerator.reduce(loss.detach(), reduction='mean')
    return loss.item()


def main(args):
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
    accelerator = Accelerator(
        mixed_precision="fp16" if args.amp else None,
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True                # 启用可重现的数据采样，保证实验的可复现性
        ),
        kwargs_handlers=[kwargs],
        step_scheduler_with_optimizer=False
    )

    trainset = SceneFlow_Dataset(split='train_finalpass', training=True)
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True, 
        drop_last=True
    )

    model = IGEVStereo()
    model.train()
    model.freeze_bn()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, 
        max_lr=args.lr,
        total_steps=len(train_loader) * args.epochs,
        pct_start=0.1,
        anneal_strategy='linear',
        cycle_momentum=False
    )

    train_loader, model, optimizer, scheduler = accelerator.prepare(train_loader, model, optimizer, scheduler)
    model.to(accelerator.device)

    for epoch in range(args.epochs):
        data_loader_with_progress = (
            tqdm(train_loader, desc=f'Ep.{epoch}/{args.epochs}', ncols=100,)
            if accelerator.is_main_process else train_loader
        )

        total_loss = 0

        for idx, data in enumerate(data_loader_with_progress):
            loss = train_iteration(model, data, optimizer, scheduler, accelerator, args)
            total_loss += loss

            if accelerator.is_main_process:
                data_loader_with_progress.set_description(
                    f'Epoch {epoch}/{args.epochs}, current loss {loss:.4f}, average loss {total_loss / (idx + 1):.4f}'
                )

        if accelerator.is_main_process:
            checkpoint = {
                'epoch': epoch,
                'model': accelerator.unwrap_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
            }
            torch.save(checkpoint, f"{args.output_dir}/checkpoint_epoch_{epoch:04d}.pth")



if __name__ == "__main__":
    args = parse_args()
    main(args)


## accelerate launch --gpu_ids="0,1,2,3"  --multi_gpu --num_processes=4 train_accelerate.py  --amp --sync-bn --output-dir ./checkpoint_accelerate