import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from pathlib import Path
import time
import random
import numpy as np
from tqdm import tqdm
import math


class Trainer:
    """分布式数据并行(DDP)训练封装类"""
    
    def __init__(self, config):
        """
        初始化 DDP 训练器
        
        参数:
            config: 包含训练配置的字典或类似对象
        """
        self.config = config
        
        # 获取分布式训练环境信息
        self.local_rank = int(os.environ.get('LOCAL_RANK', -1))
        self.world_size = int(os.environ.get('WORLD_SIZE', -1))
        self.global_rank = int(os.environ.get('RANK', -1))
        
        # 设置设备
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.local_rank)
        else:
            self.device = torch.device('cpu')
            self.local_rank = -1  # 不使用 DDP 的标志
            
        # 设置随机种子以保证可复现性
        self.set_seed(config.seed if hasattr(config, 'seed') else 42)
        
        # 初始化分布式环境(如果在分布式模式下)
        if self.is_distributed():
            self.setup_distributed()

        self.start_epoch = 0
        self.current_epoch = 0
        self.current_iteration = 0

        self.max_disp = self.config.max_disp if hasattr(self.config, 'max_disp') else 192

             
    def is_distributed(self):
        """检查是否在分布式模式下运行"""
        return self.local_rank != -1
    

    def is_main_process(self):
        """检查是否为主进程(用于日志记录和保存检查点)"""
        return self.local_rank == 0 or not self.is_distributed()
    
    
    def setup_distributed(self):
        """设置分布式环境"""
        # 初始化进程组
        if not dist.is_initialized():
            if hasattr(self.config, 'dist_backend'):
                backend = self.config.dist_backend
            else:
                backend = 'nccl' if torch.cuda.is_available() else 'gloo'
                
            dist.init_process_group(backend=backend)
            
        self.world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()
        
        if self.is_main_process():
            print(f"分布式初始化完成: 世界大小={self.world_size}, 全局等级={self.global_rank}")

    
    def set_seed(self, base_seed):
        """设置随机种子确保可复现性，但在各个进程间保持多样性"""
        # 为每个进程设置不同的种子
        seed = base_seed + self.global_rank if self.is_distributed() else base_seed
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        # 确保结果可复现
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        if self.is_main_process():
            print(f"随机种子设置为 {seed} (基础种子: {base_seed})")

    
    def prepare_model(self, model):
        """准备用于分布式训练的模型"""
        model = model.to(self.device)
        
        # 对于分布式训练，包装模型为 DDP
        if self.is_distributed():
            # 同步 BN (可选)
            if hasattr(self.config, 'sync_bn') and self.config.sync_bn:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            
            # DDP 包装
            model = DDP(
                model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.config.find_unused_parameters if hasattr(self.config, 'find_unused_parameters') else False
            )
            
        return model
    

    def prepare_dataloader(self, dataset, batch_size, shuffle=True, **kwargs):
        """创建分布式数据加载器"""
        # 对于分布式训练，使用 DistributedSampler
        if self.is_distributed():
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=shuffle
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                **kwargs
            )
        else:
            # 非分布式模式
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                **kwargs
            )
            
        return dataloader
    
    
    def save_checkpoint(self, model, optimizer, scheduler=None, scaler=None, filename=None):
        """保存检查点(仅在主进程)"""
        if not self.is_main_process():
            return
            
        # 默认文件名
        if filename is None:
            save_dir = Path(self.config.output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = save_dir / f"checkpoint_epoch_{self.current_epoch:04d}.pth"
            
        # 准备检查点数据
        state = {
            'epoch': self.current_epoch,
            'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
        if scheduler is not None:
            state['scheduler'] = scheduler.state_dict()
        if scaler is not None:
            state['scaler'] = scaler.state_dict()
                        
        # 保存检查点
        torch.save(state, filename)

            
    def load_checkpoint(self, model, optimizer=None, scheduler=None, scaler=None, filename=None):
        """加载检查点"""
        if filename is None or not os.path.isfile(filename):
            if self.is_main_process():
                print(f"未找到检查点，从头开始训练")
            return
            
        # 加载检查点
        checkpoint = torch.load(filename, map_location=self.device)
        
        # 加载模型权重
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
            
        # 加载其他状态
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            
        # 更新训练状态
        self.start_epoch = checkpoint.get('epoch', 0)
        self.current_epoch = self.start_epoch
        
        if self.is_main_process():
            print(f"已从'{filename}'加载检查点 (轮次 {self.start_epoch})")
            
        return
    
            
    def train(self, model, train_loader, optimizer, scheduler=None):
        """训练循环"""

        # 准备混合精度训练
        scaler = torch.amp.GradScaler('cuda') if self.config.amp else None
        
        # 主训练循环
        for epoch in range(self.start_epoch, self.config.epochs):
            self.current_epoch = epoch
            
            # 设置分布式采样器的轮次(确保每轮数据不同)
            if self.is_distributed() and hasattr(train_loader, 'sampler'):
                train_loader.sampler.set_epoch(epoch)
                
            # 训练一个轮次
            data_loader_with_progress = (
                tqdm(train_loader, 
                    desc=f'Ep.{epoch}/{self.config.epochs}', 
                    ncols=100,
                ) if self.is_main_process() else train_loader
            )

            total_loss = 0

            for idx, data in enumerate(data_loader_with_progress):
                loss = self._train_iteration(model, data, optimizer, scheduler, scaler)
                total_loss += loss

                # 更新进度条描述
                if self.is_main_process():
                    data_loader_with_progress.set_description(
                        f'Epoch {epoch}/{self.config.epochs}, current loss {loss:.4f}, average loss {total_loss / (idx + 1):.4f}'
                    )

                self.current_iteration += 1

            torch.cuda.synchronize()

            # 定期保存检查点
            if (epoch + 1) % self.config.save_every == 0 if hasattr(self.config, 'save_every') else False:
                self.save_checkpoint(
                    model, optimizer, scheduler, scaler,
                )
            
        # 清理分布式环境
        if self.is_distributed():
            dist.destroy_process_group()

    
    def _train_iteration(self, model, data, optimizer, scheduler, scaler):
        """训练一个迭代, IGEVStereo"""
        optimizer.zero_grad()

        left = data['left'].to(self.device, non_blocking=True)
        right = data['right'].to(self.device, non_blocking=True)
        gt_disp = data['gt_disp'].to(self.device, non_blocking=True).squeeze()

        mask = (gt_disp > 0) * (gt_disp < self.max_disp - 1)

        with torch.amp.autocast('cuda', enabled=self.config.amp):
            init_disp, disp_preds = model(left, right)

            n_predictions = len(disp_preds)
            loss = F.smooth_l1_loss(init_disp.squeeze()[mask], gt_disp[mask], reduction='mean')
            for i in range(n_predictions):
                loss += F.smooth_l1_loss(disp_preds[i].squeeze()[mask], gt_disp[mask], reduction='mean') / (2**i)

        if scaler is None:
            loss.backward()
            if self.config.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_grad)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if self.config.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

        return loss.item()