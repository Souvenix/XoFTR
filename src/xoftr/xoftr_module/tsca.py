import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange


class TextureExtractor(nn.Module):
    """纹理特征提取器
    用于从图像特征中提取纹理信息，特别适用于可见光图像的丰富纹理细节
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TextureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size, 
                               padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size, 
                               padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 高频增强滤波器
        self.high_pass_filter = nn.Conv2d(in_channels, in_channels, 3, 
                                         padding=1, bias=False)
        # 初始化高通滤波器核
        with torch.no_grad():
            kernel = torch.tensor([[-1, -1, -1],
                                 [-1,  8, -1],
                                 [-1, -1, -1]], dtype=torch.float32)
            kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(in_channels, in_channels, 1, 1)
            self.high_pass_filter.weight.copy_(kernel)
            self.high_pass_filter.weight.requires_grad = False
    
    def forward(self, x):
        # 提取高频纹理信息
        high_freq = self.high_pass_filter(x)
        
        # 通过卷积层进一步提取纹理特征
        tex_feat = self.relu(self.bn1(self.conv1(high_freq)))
        tex_feat = self.bn2(self.conv2(tex_feat))
        
        return self.relu(tex_feat)


class SemanticExtractor(nn.Module):
    """语义特征提取器
    用于从图像特征中提取语义信息，适用于两种模态的语义区域识别
    """
    def __init__(self, in_channels, out_channels):
        super(SemanticExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 全局平均池化用于捕获全局语义信息
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_channels, out_channels)
        
    def forward(self, x):
        # 局部语义特征
        local_feat = self.relu(self.bn1(self.conv1(x)))
        local_feat = self.bn2(self.conv2(local_feat))
        
        # 全局语义特征
        global_feat = self.global_pool(local_feat)
        global_feat = global_feat.view(global_feat.size(0), -1)
        global_feat = self.fc(global_feat).unsqueeze(-1).unsqueeze(-1)
        
        # 融合局部和全局语义特征
        semantic_feat = self.relu(local_feat + global_feat.expand_as(local_feat))
        
        return semantic_feat


class TextureSemanticGuidance(nn.Module):
    """纹理-语义引导权重生成模块
    结合纹理特征和语义特征生成引导权重，用于调节注意力
    """
    def __init__(self, tex_channels, sem_channels, out_channels):
        super(TextureSemanticGuidance, self).__init__()
        self.tex_proj = nn.Conv2d(tex_channels, out_channels, 1, bias=False)
        self.sem_proj = nn.Conv2d(sem_channels, out_channels, 1, bias=False)
        
        # 融合网络
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 权重生成
        self.weight_gen = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, tex_feat, sem_feat):
        # 投影到统一维度
        tex_proj = self.tex_proj(tex_feat)
        sem_proj = self.sem_proj(sem_feat)
        
        # 融合纹理和语义特征
        fusion_feat = torch.cat([tex_proj, sem_proj], dim=1)
        fusion_feat = self.fusion_conv(fusion_feat)
        
        # 生成引导权重
        guidance_weight = self.weight_gen(fusion_feat)
        
        return guidance_weight


class GuidedCrossAttention(nn.Module):
    """引导交叉注意力模块
    使用纹理-语义引导权重调节交叉注意力计算
    """
    def __init__(self, d_model, nhead, guidance_channels):
        super(GuidedCrossAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.merge = nn.Linear(d_model, d_model, bias=False)
        
        # 引导权重投影
        self.guidance_proj = nn.Conv2d(1, nhead, 1, bias=False)
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, guidance_weight, q_mask=None, kv_mask=None):
        """
        Args:
            query: [N, L, C]
            key: [N, S, C]
            value: [N, S, C]
            guidance_weight: [N, 1, H, W] - 引导权重
            q_mask: [N, L] (optional)
            kv_mask: [N, S] (optional)
        """
        bs = query.size(0)
        
        # 线性投影
        q = self.q_proj(query).view(bs, -1, self.nhead, self.dim)
        k = self.k_proj(key).view(bs, -1, self.nhead, self.dim)
        v = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        
        # 重塑为2D特征图以应用引导权重
        # 自动计算特征图尺寸，确保 L = H * W
        def calc_hw(total_pixels):
            h = int(total_pixels ** 0.5)
            w = total_pixels // h
            while h * w != total_pixels:
                h -= 1
                w = total_pixels // h
            return h, w
        
        hw_q = calc_hw(query.size(1))
        hw_kv = calc_hw(key.size(1))
        
        # 计算注意力分数
        scores = torch.einsum('nlhd,nshd->nhls', q, k) / (self.dim ** 0.5)
        
        # 应用引导权重调节
        if guidance_weight is not None:
            # 投影引导权重到注意力头数
            guidance_proj = self.guidance_proj(guidance_weight)  # [N, nhead, H, W]
            
            # 调整尺寸匹配注意力分数
            if guidance_proj.size(-1) != hw_kv[1] or guidance_proj.size(-2) != hw_kv[0]:
                guidance_proj = F.interpolate(guidance_proj, size=hw_kv, 
                                          mode='bilinear', align_corners=False)
            
            # 重塑为注意力权重格式 - 注意scores的维度是[N, nhead, L, S]
            # 我们需要为每个query位置提供引导权重，所以重塑为[N, nhead, 1, S]
            guidance_weight_flat = guidance_proj.view(bs, self.nhead, 1, -1)  # [N, nhead, 1, S]
            
            # 应用引导权重到注意力分数
            scores = scores * (1 + guidance_weight_flat.expand_as(scores))
        
        # 应用mask
        if q_mask is not None:
            scores = scores.masked_fill(q_mask.unsqueeze(1).unsqueeze(-1), float('-inf'))
        if kv_mask is not None:
            scores = scores.masked_fill(kv_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力
        out = torch.einsum('nhls,nshd->nlhd', attn_weights, v)
        out = out.contiguous().view(bs, -1, self.d_model)
        
        # 合并多头
        out = self.merge(out)
        
        return self.norm(out)


class TSCAModule(nn.Module):
    """纹理语义特征引导交叉注意力模块(TSCA)
    
    核心思想：在计算跨模态相似度之前，引入由纹理特征与语义特征联合形成的引导权重，
    对注意力权值进行调节，使模型在特征匹配过程中更关注两种模态间纹理结构与语义区域的一致性。
    """
    def __init__(self, config):
        super(TSCAModule, self).__init__()
        
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        
        # 纹理特征提取器
        self.texture_extractor = TextureExtractor(
            in_channels=self.d_model,
            out_channels=self.d_model
        )
        
        # 语义特征提取器
        self.semantic_extractor = SemanticExtractor(
            in_channels=self.d_model,
            out_channels=self.d_model
        )
        
        # 纹理-语义引导权重生成
        self.guidance_generator = TextureSemanticGuidance(
            tex_channels=self.d_model,
            sem_channels=self.d_model,
            out_channels=self.d_model
        )
        
        # 引导交叉注意力
        self.guided_cross_attn = GuidedCrossAttention(
            d_model=self.d_model,
            nhead=self.nhead,
            guidance_channels=1
        )
        
        # 前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(self.d_model * 2, self.d_model, bias=False),
        )
        
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        
    def forward(self, feat0, feat1, hw_size=None, mask0=None, mask1=None):
        """
        Args:
            feat0: [N, L, C] - 第一个模态的特征
            feat1: [N, S, C] - 第二个模态的特征
            hw_size: (H, W) - 特征图的空间尺寸
            mask0: [N, L] (optional) - 第一个模态的mask
            mask1: [N, S] (optional) - 第二个模态的mask
        """
        if hw_size is None:
            # 自动计算特征图尺寸，确保 L = H * W
            total_pixels = feat0.size(1)
            h = int(total_pixels ** 0.5)
            w = total_pixels // h
            # 确保 h * w == total_pixels
            while h * w != total_pixels:
                h -= 1
                w = total_pixels // h
            hw_size = (h, w)
        
        # 重塑为2D特征图
        feat0_2d = feat0.transpose(1, 2).view(feat0.size(0), -1, hw_size[0], hw_size[1])
        
        # 为feat1计算合适的尺寸（可能与feat0不同）
        if feat1.size(1) != feat0.size(1):
            total_pixels1 = feat1.size(1)
            h1 = int(total_pixels1 ** 0.5)
            w1 = total_pixels1 // h1
            while h1 * w1 != total_pixels1:
                h1 -= 1
                w1 = total_pixels1 // h1
            hw_size1 = (h1, w1)
        else:
            hw_size1 = hw_size
            
        feat1_2d = feat1.transpose(1, 2).view(feat1.size(0), -1, hw_size1[0], hw_size1[1])
        
        # 提取纹理特征
        tex_feat0 = self.texture_extractor(feat0_2d)
        tex_feat1 = self.texture_extractor(feat1_2d)
        
        # 提取语义特征
        sem_feat0 = self.semantic_extractor(feat0_2d)
        sem_feat1 = self.semantic_extractor(feat1_2d)
        
        # 生成引导权重
        guidance_weight0 = self.guidance_generator(tex_feat0, sem_feat0)
        guidance_weight1 = self.guidance_generator(tex_feat1, sem_feat1)
        
        # 如果两个特征图的尺寸不同，需要调整引导权重的尺寸
        if hw_size1 != hw_size and guidance_weight1.size() != guidance_weight0.size():
            # 调整guidance_weight1的尺寸以匹配guidance_weight0
            guidance_weight1 = F.interpolate(guidance_weight1, size=guidance_weight0.shape[2:], 
                                           mode='bilinear', align_corners=False)
        
        # 应用引导交叉注意力
        # feat0 关注 feat1
        cross_feat0 = self.guided_cross_attn(
            feat0, feat1, feat1, guidance_weight1,
            q_mask=mask0, kv_mask=mask1
        )
        
        # feat1 关注 feat0
        cross_feat1 = self.guided_cross_attn(
            feat1, feat0, feat0, guidance_weight0,
            q_mask=mask1, kv_mask=mask0
        )
        
        # 前馈网络
        feat0_out = self.norm1(feat0 + cross_feat0)
        feat1_out = self.norm1(feat1 + cross_feat1)
        
        # MLP
        feat0_mlp = self.mlp(torch.cat([feat0_out, cross_feat0], dim=2))
        feat1_mlp = self.mlp(torch.cat([feat1_out, cross_feat1], dim=2))
        
        feat0_final = self.norm2(feat0_out + feat0_mlp)
        feat1_final = self.norm2(feat1_out + feat1_mlp)
        
        return feat0_final, feat1_final
    
    def get_regularization_loss(self):
        """获取TSCA模块的正则化损失"""
        reg_loss = 0.0
        
        # 对纹理和语义提取器的参数进行L2正则化
        for module in [self.texture_extractor, self.semantic_extractor, self.guidance_generator]:
            for param in module.parameters():
                reg_loss += torch.norm(param, p=2)
        
        # 对引导交叉注意力的参数进行L2正则化
        for param in self.guided_cross_attn.parameters():
            reg_loss += torch.norm(param, p=2)
        
        return reg_loss * 1e-6  # 使用较小的权重