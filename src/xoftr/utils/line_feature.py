import torch
import torch.nn as nn
import numpy as np
import cv2
from pytlsd import lsd
import os

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
        
        # 添加基于配置的融合权重参数（非学习的）
        self.fusion_weight = config.get('line_fusion', {}).get('weight', 0.5)  # 默认为0.5
        
        # 线质量阈值参数
        self.length_threshold = config.get('line_fusion', {}).get('length_threshold', 10.0)  # 线长度阈值
        self.quality_threshold = config.get('line_fusion', {}).get('quality_threshold', 0.8)  # 线质量阈值
        
        # 光照条件阈值
        self.light_threshold = 0.3  # 光照亮度阈值，低于此值认为光照条件较差
    
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
            
            # 简单评估光照条件（非学习的）
            mean_intensity = np.mean(img_np) / 255.0
            is_low_light = mean_intensity < self.light_threshold
            
            # 使用LSD算法检测线
            lines = lsd(img_np)
            
            # 创建线特征图，应用质量筛选
            line_map = np.zeros_like(img_np, dtype=np.float32)
            if lines is not None and len(lines) > 0:
                for line in lines:
                    x1, y1, x2, y2 = map(int, line[:4])
                    
                    # 计算线的长度作为质量指标之一
                    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)** 2)
                    
                    # 获取LSD返回的质量信息（如果可用）
                    quality = line[5] if len(line) > 5 else 1.0
                    
                    # 应用质量筛选
                    # 在低光照条件下，提高质量阈值，更严格地筛选线特征
                    current_quality_threshold = self.quality_threshold * (1.0 + 0.5 * is_low_light)
                    
                    if line_length >= self.length_threshold and quality >= current_quality_threshold:
                        # 根据线长度和质量调整线的强度
                        intensity = min(1.0, quality * min(1.0, line_length / 100.0))
                        
                        # 在低光照条件下，降低线特征的整体强度
                        if is_low_light:
                            intensity *= 0.7  # 降低30%
                        
                        # 绘制线特征
                        thickness = 1  # 保持固定线宽
                        cv2.line(line_map, (x1, y1), (x2, y2), intensity, thickness=thickness)
            
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
        # 使用手工设计的融合权重
        # 计算线特征的统计信息，用于动态调整权重
        line_mean = torch.mean(line_feat, dim=(1, 2, 3), keepdim=True)
        
        # 根据线特征的显著性动态调整融合权重
        # 线特征越明显（均值越高），权重越大
        dynamic_weight = self.fusion_weight * torch.clamp(line_mean * 5.0, 0.3, 1.0)
        
        # 应用动态权重到线特征
        weighted_line_feat = line_feat * dynamic_weight
        
        # 特征拼接
        combined = torch.cat([image_feat, weighted_line_feat], dim=1)
        # 通过1x1卷积融合特征
        fused_feat = self.fusion_module(combined)
        # 跳跃连接
        fused_feat = fused_feat + image_feat
        
        return fused_feat
        
    def visualize_and_save_lines(self, image, save_dir='./output_lines', file_prefix='line_detection'):
        """
        将检测到的线标注在原始图像上并保存图片
        Args:
            image: [N, 1, H, W] 输入图像
            save_dir: 保存图像的目录
            file_prefix: 保存图像的文件名前缀
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        N, C, H, W = image.shape
        
        for i in range(N):
            # 将图像归一化到0-255范围
            img_np = image[i, 0].detach().cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255
            img_np = img_np.astype(np.uint8)
            
            # 如果是灰度图，转换为RGB以支持彩色线条
            if len(img_np.shape) == 2:
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = img_np.copy()
            
            # 评估光照条件
            mean_intensity = np.mean(img_np) / 255.0
            is_low_light = mean_intensity < self.light_threshold
            
            # 使用LSD算法检测线
            lines = lsd(img_np)
            
            # 在图像上绘制检测到的线（使用红色）
            if lines is not None and len(lines) > 0:
                for line in lines:
                    x1, y1, x2, y2 = map(int, line[:4])
                    
                    # 计算线的长度和质量，仅绘制高质量的线
                    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)** 2)
                    quality = line[5] if len(line) > 5 else 1.0
                    
                    current_quality_threshold = self.quality_threshold * (1.0 + 0.5 * is_low_light)
                    
                    if line_length >= self.length_threshold and quality >= current_quality_threshold:
                        # 根据线质量调整线条粗细
                        thickness = max(1, int(quality * 2))
                        cv2.line(img_rgb, (x1, y1), (x2, y2), (0, 0, 255), thickness=thickness)
            
            # 保存标注后的图像
            save_path = os.path.join(save_dir, f'{file_prefix}_{i}.png')
            cv2.imwrite(save_path, img_rgb)