# src/model.py
import torch
from torch import nn
import timm
from src.cbam import CBAM
import torch.nn.functional as F

class DRMultiTaskModel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', pretrained=True, n_iqa_classes=2, drop_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='')
        # get feature dim by forwarding a dummy
        # but timm models usually have feature_info; simpler: use classifier features
        feature_dim = self.backbone.num_features if hasattr(self.backbone, 'num_features') else 1280
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cbam = CBAM(feature_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.fc_dr = nn.Linear(feature_dim, 1)   # binary referable
        self.fc_iqa = nn.Linear(feature_dim, n_iqa_classes)  # IQA classes

        # initialize heads
        nn.init.normal_(self.fc_dr.weight, 0, 0.01)
        nn.init.constant_(self.fc_dr.bias, 0)
        nn.init.normal_(self.fc_iqa.weight, 0, 0.01)
        nn.init.constant_(self.fc_iqa.bias, 0)

    def forward(self, x):
        # backbone returns feature map [B, C, H, W]
        feats = self.backbone.forward_features(x)  # timm convention
        feats = self.cbam(feats)
        pooled = self.global_pool(feats).flatten(1)
        pooled = self.dropout(pooled)
        dr_logit = self.fc_dr(pooled).squeeze(1)   # shape [B]
        iqa_logits = self.fc_iqa(pooled)           # shape [B, n_iqa]
        return dr_logit, iqa_logits
