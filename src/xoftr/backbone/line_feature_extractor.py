import torch
import torch.nn as nn
import torch.nn.functional as F

class LineFeatureExtractor(nn.Module):
    """用于提取线特征的网络模块"""
    def __init__(self, config):
        super().__init__()
        # 使用边缘检测网络提取线特征
        self.edge_detection = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        )
        
        # 线特征增强模块
        self.line_enhancer = nn.Sequential(
            nn.Conv2d(1, config['initial_dim'], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(config['initial_dim']),
            nn.ReLU(inplace=True)
        )
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: [N, 1, H, W] 输入图像
        
        # 提取边缘特征
        edges = self.edge_detection(x)
        edges = torch.sigmoid(edges)  # 将输出压缩到[0, 1]范围
        
        # 增强线特征
        line_feats = self.line_enhancer(edges)
        
        return line_feats, edges