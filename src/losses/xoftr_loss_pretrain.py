import torch
import torch.nn as nn
import torch.nn.functional as F

class XoFTRLossPretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.W_f = config["xoftr"]['fine_window_size']
    
    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}

        pred0, pred1 = data["pred0"], data["pred1"]
        target0, target1 = data["target0"], data["target1"]
        target0 = target0[[data['b_ids'], data['i_ids']]]
        target1 = target1[[data['b_ids'], data['j_ids']]]
        
        # get correct indices
        pred0 = pred0[data["ids_image0"]]
        pred1 = pred1[data["ids_image1"]]
        target0 = target0[data["ids_image0"]]
        target1 = target1[data["ids_image1"]]
        
        loss0 = (pred0 - target0)**2
        loss1 = (pred1 - target1)**2
        l_mod = self.modality_alignment_loss(data['modality_feats'])

        loss = loss0.mean() + loss1.mean() + 0.05 * l_mod
        
        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})

    def modality_alignment_loss(self, modality_feats, mask=None):
        """
        modality_feats: dict {'c':(m0,m1), 'm':..., 'f':...}
        """
        loss = 0.0
        for level in modality_feats:
            m0, m1 = modality_feats[level]
            loss += F.mse_loss(m0, m1)

        return loss / len(modality_feats)