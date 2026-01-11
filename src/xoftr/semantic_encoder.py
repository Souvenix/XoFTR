import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticEncoder(nn.Module):
    """
    共享参数的语义编码器
    输入：image (N,1,H,W) or (N,3,H,W)
    输出：S ∈ [N, C_s, H/8, W/8]
    """
    def __init__(self, in_ch=1, base_dim=64, out_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, base_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_dim, base_dim*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_dim*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_dim*2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)

class SemanticProjection(nn.Module):
    """
    将语义特征映射到与 coarse feature 相同维度
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, S):
        # S: [N, HW, C_s]
        return self.proj(S)

class SemanticGuidanceModule(nn.Module):
    def __init__(self, temperature=0.07, normalize=True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, S0, S1):
        """
        S0, S1: [N, HW, C]
        输出：
          B_sem: [N, HW0, HW1]
        """
        if self.normalize:
            S0 = F.normalize(S0, dim=-1)
            S1 = F.normalize(S1, dim=-1)

        sim = torch.matmul(S0, S1.transpose(1, 2))  # cosine sim
        B_sem = sim / self.temperature
        return B_sem