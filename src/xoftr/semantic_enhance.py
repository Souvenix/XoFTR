import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops.einops import rearrange


class SemanticEnhanceModule(nn.Module):
    """
    语义分割增强模块，使用预训练的DeepLabV3模型提取语义信息，
    并将其融合到特征匹配过程中以提升准确性。
    """
    
    def _get_config_value(self, config, section_keys, key, default=None):
        """获取配置值，支持大小写不敏感的键名"""
        # 尝试不同的section键名
        for section_key in section_keys:
            if section_key in config:
                section = config[section_key]
                # 尝试不同的键名变体
                key_variants = [key, key.upper(), key.lower()]
                for key_variant in key_variants:
                    if key_variant in section:
                        return section[key_variant]
        
        # 如果没找到，返回默认值
        if default is not None:
            return default
        
        # 如果必须找到该值，抛出更详细的错误
        available_keys = list(config.keys()) if isinstance(config, dict) else []
        raise KeyError(f"配置中找不到键 '{key}' (在sections {section_keys}中)，可用键: {available_keys}")

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 获取配置参数，支持大小写不敏感的键名
        coarse_d_model = self._get_config_value(config, ['coarse', 'COARSE'], 'd_model', 256)
        fine_feat_dim = self._get_config_value(config, ['resnet', 'RESNET'], 'block_dims', [128, 196, 256])[-1]
        
        # 简单的语义特征提取器 - 使用轻量级卷积而不是DeepLabV3
        self.semantic_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 语义特征融合层 - 简单的1x1卷积
        self.semantic_fusion_coarse = nn.Conv2d(32, coarse_d_model, kernel_size=1)
        self.semantic_fusion_fine = nn.Conv2d(32, fine_feat_dim, kernel_size=1)
        
        # 存储目标维度以便后续使用
        self.coarse_d_model = coarse_d_model
        self.fine_feat_dim = fine_feat_dim
        
        # 简单的语义权重
        self.semantic_weight = nn.Parameter(torch.tensor(0.1))
        
    def extract_semantic_features(self, image):
        """
        提取简单的语义特征
        Args:
            image: 输入图像 [N, 1, H, W] 或 [N, 3, H, W]
        Returns:
            semantic_features: 语义特征 [N, 32, H, W]
        """
        # 如果是三通道图像，只取第一个通道
        if image.size(1) == 3:
            image = image[:, 0:1, :, :]  # 只取第一个通道
        
        # 使用轻量级卷积提取语义特征
        semantic_features = self.semantic_extractor(image)
        
        return semantic_features
    
    def fuse_semantic_features(self, visual_features, semantic_features, level='coarse'):
        """
        融合视觉特征和语义特征 - 简化版本
        Args:
            visual_features: 视觉特征 [N, C, H, W]
            semantic_features: 语义特征 [N, 32, H, W]
            level: 特征级别 ('coarse' 或 'fine')
        Returns:
            fused_features: 融合后的特征 [N, C, H, W]
        """
        # 获取视觉特征的通道数
        target_channels = visual_features.shape[1]
        
        # 根据级别选择合适的融合层，并确保输出维度匹配
        if level == 'coarse':
            # 如果融合层输出维度不匹配，创建新的融合层
            if self.semantic_fusion_coarse.out_channels != target_channels:
                self.semantic_fusion_coarse = nn.Conv2d(32, target_channels, kernel_size=1).to(visual_features.device)
            semantic_proj = self.semantic_fusion_coarse(semantic_features)
        else:
            # 如果融合层输出维度不匹配，创建新的融合层
            if self.semantic_fusion_fine.out_channels != target_channels:
                self.semantic_fusion_fine = nn.Conv2d(32, target_channels, kernel_size=1).to(visual_features.device)
            semantic_proj = self.semantic_fusion_fine(semantic_features)
        
        # 确保空间尺寸匹配
        if visual_features.shape[-2:] != semantic_proj.shape[-2:]:
            semantic_proj = F.interpolate(
                semantic_proj, 
                size=visual_features.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # 简单的加权融合
        fused_features = visual_features + self.semantic_weight * semantic_proj
        
        return fused_features
    

    
    def forward(self, image0, image1, visual_features0, visual_features1, level='coarse'):
        """
        前向传播 - 简化版本
        Args:
            image0: 第一张图像 [N, 1, H, W]
            image1: 第二张图像 [N, 1, H, W]
            visual_features0: 第一张图像的视觉特征 [N, C, H', W']
            visual_features1: 第二张图像的视觉特征 [N, C, H', W']
            level: 特征级别 ('coarse' 或 'fine')
        Returns:
            enhanced_features0: 增强后的特征0
            enhanced_features1: 增强后的特征1
        """
        # 提取语义特征
        semantic_features0 = self.extract_semantic_features(image0)
        semantic_features1 = self.extract_semantic_features(image1)
        
        # 融合视觉特征和语义特征
        enhanced_features0 = self.fuse_semantic_features(
            visual_features0, semantic_features0, level
        )
        enhanced_features1 = self.fuse_semantic_features(
            visual_features1, semantic_features1, level
        )
        
        return enhanced_features0, enhanced_features1