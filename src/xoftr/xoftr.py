import time
import torch
import torch.nn as nn
from einops.einops import rearrange
from .backbone import ResNet_8_2
from .utils.position_encoding import PositionEncodingSine
from .xoftr_module import LocalFeatureTransformer, FineProcess, CoarseMatching, FineSubMatching
from .utils.line_feature import LineFeatureExtractor  # 导入线特征提取器

class XoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = ResNet_8_2(config['resnet'])
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_process = FineProcess(config)
        self.fine_matching = FineSubMatching(config)
        self.line_feature_extractor = LineFeatureExtractor(config)  # 添加线特征提取器


    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        eps = 1e-6

        image0_mean = data['image0'].mean(dim=[2,3], keepdim=True)
        image0_std = data['image0'].std(dim=[2,3], keepdim=True)
        image0 = (data['image0'] - image0_mean) / (image0_std + eps)

        image1_mean = data['image1'].mean(dim=[2,3], keepdim=True)
        image1_std = data['image1'].std(dim=[2,3], keepdim=True)
        image1 = (data['image1'] - image1_mean) / (image1_std + eps)

        # start_time = time.time()
        # 提取线特征
        line_feat0 = self.line_feature_extractor(data['image0'])
        line_feat1 = self.line_feature_extractor(data['image1'])

        # t1 = time.time()
        # print(f"提取线特征耗时: {t1 - start_time:.4f} 秒")

        if False:
            self.line_feature_extractor.visualize_and_save_lines(
                data['image0'], 
                save_dir='./output_lines/image0', 
                file_prefix='image0_lines'
            )
            self.line_feature_extractor.visualize_and_save_lines(
                data['image1'], 
                save_dir='./output_lines/image1', 
                file_prefix='image1_lines'
            )

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_m0, feat_m1) = feats_m.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
        else:  # handle different input shapes
            feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1 = self.backbone(image1)

        # t2 = time.time()
        # print(f"Local CNN耗时: {t2 - start_time:.4f} 秒")

        # 融合线特征和图像特征
        feat_f0 = self.line_feature_extractor.fuse_features(feat_f0, line_feat0)
        feat_f1 = self.line_feature_extractor.fuse_features(feat_f1, line_feat1)

        # t3 = time.time()
        # print(f"融合耗时: {t3 - t2:.4f} 秒")

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_m': feat_m0.shape[2:], 'hw1_m': feat_m1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # save coarse features for fine matching
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # t4 = time.time()
        # print(f"coarse-level loftr耗时: {t4 - t3:.4f} 秒")

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # t5 = time.time()
        # print(f"match coarse-level耗时: {t5 - t4:.4f} 秒")

        # 4. fine-level matching module       
        feat_f0_unfold, feat_f1_unfold = self.fine_process(feat_f0, feat_f1,
                                                           feat_m0, feat_m1,
                                                           feat_c0, feat_c1,
                                                           feat_c0_pre, feat_c1_pre,
                                                           data) 

        # 5. match fine-level and sub-pixel refinement
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

        # t6 = time.time()
        # print(f"match fine-level耗时: {t6- t5:.4f} 秒")

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        # 设置strict=False以忽略新增的线特征模块参数
        return super().load_state_dict(state_dict, strict=False, *args, **kwargs)
