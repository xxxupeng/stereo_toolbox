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


from stereo_toolbox.datasets import *
from stereo_toolbox.models import *
from stereo_toolbox.visualization import *
from stereo_toolbox.trainer import Trainer


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

    return parser.parse_args()


def main():
    args = parse_args()

    trainer = Trainer(args)

    trainset = SceneFlow_Dataset(split='train_finalpass', training=True)
    train_loader = trainer.prepare_dataloader(
        dataset=trainset, 
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=True,
        shuffle=True,
    )

    model = IGEVStereo()
    model.train()
    model.freeze_bn()
    model = trainer.prepare_model(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, 
        max_lr=args.lr,
        total_steps=len(train_loader) * args.epochs,
        pct_start=0.1,
        anneal_strategy='linear',
        cycle_momentum=False
    )

    trainer.train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        scheduler=scheduler
    )


if __name__ == "__main__":
    main()


## torchrun --nproc_per_node=4 train.py  --amp --sync-bn 