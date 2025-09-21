import torch
import torch.nn as nn
import numpy as np
import cv2
from pytlsd import lsd

class LineFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 配置参数
        self.config = config
        # 线特征处理的通道数
        self.line_feat_dim = config['resnet']['block_dims'][0]  # 使用与fine level相同的维度
        
        # 线特征转换模块，将线特征转换为与图像特征兼容的维度
        self.line_proj = nn.Sequential(
            nn.Conv2d(1, self.line_feat_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.line_feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.line_feat_dim // 2, self.line_feat_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.line_feat_dim),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合模块
        self.fusion_module = nn.Conv2d(self.line_feat_dim * 2, self.line_feat_dim, kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, image):
        """
        提取线特征并与原始特征融合
        Args:
            image: [N, 1, H, W] 输入图像
        Returns:
            line_feat: [N, C, H/2, W/2] 线特征图，分辨率与fine level相同
        """
        N, C, H, W = image.shape
        
        # 将tensor转换为numpy数组用于LSD线检测
        line_maps = []
        for i in range(N):
            # 将图像归一化到0-255范围
            img_np = image[i, 0].detach().cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255
            img_np = img_np.astype(np.uint8)
            
            # 使用LSD算法检测线
            lines = lsd(img_np)
            
            # 创建线特征图
            line_map = np.zeros_like(img_np, dtype=np.float32)
            if lines is not None and len(lines) > 0:
                for line in lines:
                    x1, y1, x2, y2 = map(int, line[:4])
                    cv2.line(line_map, (x1, y1), (x2, y2), 1.0, thickness=1)
            
            # 扩展维度并添加到列表
            line_map = np.expand_dims(line_map, axis=0)  # [1, H, W]
            line_maps.append(line_map)
        
        # 将线特征图转换为tensor
        line_maps = np.stack(line_maps, axis=0)  # [N, 1, H, W]
        line_maps = torch.from_numpy(line_maps).to(image.device)
        
        # 调整线特征图分辨率到与fine level相同（1/2）
        line_maps = nn.functional.interpolate(
            line_maps, 
            size=(H//2, W//2), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 通过投影模块处理线特征
        line_feat = self.line_proj(line_maps)
        
        return line_feat

    def fuse_features(self, image_feat, line_feat):
        """
        融合图像特征和线特征
        Args:
            image_feat: [N, C, H, W] 原始图像特征
            line_feat: [N, C, H, W] 线特征
        Returns:
            fused_feat: [N, C, H, W] 融合后的特征
        """
        # 特征拼接
        combined = torch.cat([image_feat, line_feat], dim=1)
        # 通过1x1卷积融合特征
        fused_feat = self.fusion_module(combined)
        # 跳跃连接
        fused_feat = fused_feat + image_feat
        
        return fused_feat