import torch
import torch.nn as nn
import numpy as np
import cv2
from pytlsd import lsd
import os


class LineFeatureExtractor(nn.Module):
    """
    LFFM: Line Feature Fusion Module (token-level, cross-attention based)

    - 线特征：线段 token (x1,y1,x2,y2)
    - 点特征：coarse-level feature map tokens
    - 融合方式：Point-Line Cross Attention + Residual
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # === 对齐 XoFTR coarse-level 维度（关键）===
        self.line_feat_dim = 128

        # ===== Line Embedding (L x 4 -> L x C) =====
        self.line_embed = nn.Sequential(
            nn.Linear(4, self.line_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.line_feat_dim, self.line_feat_dim)
        )

        # ===== Point-Line Cross Attention =====
        self.line_pl_attn = nn.MultiheadAttention(
            embed_dim=self.line_feat_dim,
            num_heads=4,
            batch_first=True
        )

        # 可学习残差权重
        self.line_alpha = nn.Parameter(torch.tensor(0.5))

        # ===== 线质量控制（保持你原来的逻辑）=====
        self.length_threshold = config.get('line_fusion', {}).get('length_threshold', 10.0)
        self.quality_threshold = config.get('line_fusion', {}).get('quality_threshold', 0.8)
        self.light_threshold = 0.3

    # ------------------------------------------------------------------
    # Line token extraction (不涉及任何 CNN / Linear)
    # ------------------------------------------------------------------
    def forward(self, image):
        """
        Args:
            image: [N, 1, H, W]
        Returns:
            line_tokens_batch: List[Tensor], 每个 Tensor 是 [L, 4]
        """
        N, _, H, W = image.shape
        device = image.device

        line_tokens_batch = []

        for i in range(N):
            img_np = image[i, 0].detach().cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255
            img_np = img_np.astype(np.uint8)

            mean_intensity = np.mean(img_np) / 255.0
            is_low_light = mean_intensity < self.light_threshold

            lines = self.extract_lines_opencv(img_np)
            line_tokens = []

            if lines is not None:
                for line in lines:
                    coords = line.flatten()[:4]
                    x1, y1, x2, y2 = map(int, coords)

                    line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    quality = line[5] if len(line) > 5 else 1.0
                    current_q = self.quality_threshold * (1.0 + 0.5 * is_low_light)

                    if line_length >= self.length_threshold and quality >= current_q:
                        # 归一化坐标
                        line_tokens.append([
                            x1 / W, y1 / H,
                            x2 / W, y2 / H
                        ])

            if len(line_tokens) == 0:
                line_tokens = [[0.0, 0.0, 0.0, 0.0]]

            line_tokens_batch.append(
                torch.tensor(line_tokens, dtype=torch.float32, device=device)
            )

        return line_tokens_batch

    # ------------------------------------------------------------------
    # LFFM core: Point-Line Cross Attention
    # ------------------------------------------------------------------
    def fuse_features(self, image_feat, line_tokens_batch):
        """
        Args:
            image_feat: [N, C, H, W]  (coarse-level feature map)
            line_tokens_batch: List[Tensor], each [L, 4]
        Returns:
            fused_feat: [N, C, H, W]
        """
        N, C, H, W = image_feat.shape
        assert C == self.line_feat_dim, \
            f"Feature dim mismatch: image_feat C={C}, line_feat_dim={self.line_feat_dim}"

        # Point tokens: [N, HW, C]
        point_tokens = image_feat.flatten(2).transpose(1, 2)

        enhanced_tokens = []

        for i in range(N):
            MAX_LINES = 64  # 32 / 64 都可以

            line_tokens = line_tokens_batch[i]
            if line_tokens.shape[0] > MAX_LINES:
                line_tokens = line_tokens[:MAX_LINES]
            line_emb = self.line_embed(line_tokens)     # [L, C]

            # Cross-Attention
            attn_out, _ = self.line_pl_attn(
                query=point_tokens[i:i + 1],            # [1, HW, C]
                key=line_emb.unsqueeze(0),              # [1, L, C]
                value=line_emb.unsqueeze(0)
            )

            enhanced = point_tokens[i] + self.line_alpha * attn_out.squeeze(0)
            enhanced_tokens.append(enhanced)

        enhanced_tokens = torch.stack(enhanced_tokens, dim=0)
        fused_feat = enhanced_tokens.transpose(1, 2).reshape(N, C, H, W)

        return fused_feat

    # ------------------------------------------------------------------
    # Visualization (保持不变)
    # ------------------------------------------------------------------
    def visualize_and_save_lines(self, image, save_dir='./output_lines', file_prefix='line_detection'):
        os.makedirs(save_dir, exist_ok=True)
        N, _, H, W = image.shape

        for i in range(N):
            img_np = image[i, 0].detach().cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255
            img_np = img_np.astype(np.uint8)

            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            mean_intensity = np.mean(img_np) / 255.0
            is_low_light = mean_intensity < self.light_threshold

            lines = lsd(img_np)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = map(int, line[:4])
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    quality = line[5] if len(line) > 5 else 1.0
                    current_q = self.quality_threshold * (1.0 + 0.5 * is_low_light)

                    if length >= self.length_threshold and quality >= current_q:
                        cv2.line(img_rgb, (x1, y1), (x2, y2), (0, 0, 255), 1)

            cv2.imwrite(os.path.join(save_dir, f'{file_prefix}_{i}.png'), img_rgb)

    def extract_lines_opencv(self, img_np):
        edges = cv2.Canny(img_np, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=30, minLineLength=20, maxLineGap=5
        )
        return lines
