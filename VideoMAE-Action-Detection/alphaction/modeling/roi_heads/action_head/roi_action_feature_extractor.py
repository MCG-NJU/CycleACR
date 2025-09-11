import torch
from torch import nn
from torch.nn import functional as F

from alphaction.modeling import registry
from alphaction.modeling.poolers import make_3d_pooler
from alphaction.modeling.roi_heads.action_head.cycleacr import make_cycleacr_module
from alphaction.modeling.utils import cat, pad_sequence, prepare_pooled_feature


@registry.ROI_ACTION_FEATURE_EXTRACTORS.register("2MLPFeatureExtractor")
class MLPFeatureExtractor(nn.Module):
    def __init__(self, config, dim_in):
        super(MLPFeatureExtractor, self).__init__()
        self.config = config
        head_cfg = config.MODEL.ROI_ACTION_HEAD

        self.pooler = make_3d_pooler(head_cfg)

        resolution = head_cfg.POOLER_RESOLUTION

        self.max_pooler = nn.MaxPool3d((1, resolution, resolution))

        if config.MODEL.CYCLE_ACR.ACTIVE:
            self.cycleacr_module = make_cycleacr_module(config, dim_in)

        representation_size = head_cfg.MLP_HEAD_DIM

        fc1_dim_in = dim_in
        if config.MODEL.CYCLE_ACR.ACTIVE and (config.MODEL.CYCLE_ACR.FUSION == "concat"):
            fc1_dim_in += config.MODEL.CYCLE_ACR.DIM_OUT

        self.fc1 = nn.Linear(fc1_dim_in, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)

        for l in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

        self.dim_out = representation_size

    def roi_pooling(self, slow_features, fast_features, proposals):
        if slow_features is not None:
            if self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                slow_features = slow_features.mean(dim=2, keepdim=True)
            slow_x = self.pooler(slow_features, proposals)
            if not self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                slow_x = slow_x.mean(dim=2, keepdim=True)
            x = slow_x
        if fast_features is not None:
            if self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                fast_features = fast_features.mean(dim=2, keepdim=True)
            fast_x = self.pooler(fast_features, proposals)
            if not self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                fast_x = fast_x.mean(dim=2, keepdim=True)
            x = fast_x

        if slow_features is not None and fast_features is not None:
            x = torch.cat([slow_x, fast_x], dim=1)
        return x

    def forward(self, slow_features, fast_features, proposals, extras={}, part_forward=-1):
        if part_forward == 1:
            local_features = None
            roi_features = cat([box.get_field("pooled_feature") for box in extras["current_feat_r"]])  # [n, dim_in, 1, 1, 1]
            actor_features = cat([box.get_field("pooled_feature") for box in proposals])  # [n, dim_inner, 1, 1, 1]

            context_features = None
        else:
            local_features = self.roi_pooling(slow_features, fast_features, proposals)  # [n, dim_in, 1, 7, 7]
            roi_features = self.max_pooler(local_features)  # [n, dim_in, 1, 1, 1]
            actor_features = roi_features

            if fast_features is None:
                context_features = slow_features
            else:
                slow_features = F.interpolate(slow_features, fast_features.shape[-3:])
                context_features = torch.cat([slow_features, fast_features], dim=1)
            context_features = nn.AdaptiveMaxPool3d((context_features.shape[2], 1, 1))(context_features)  # [b, dim_in, t_fast, 1, 1]

        final_features = None

        if hasattr(self, "cycleacr_module"):
            final_features, actor_features = self.cycleacr_module(actor_features, proposals, context_features, local_features, extras, part_forward)

        if part_forward == 0:
            return None, actor_features, roi_features

        x_after = self.fusion(roi_features, final_features, self.config.MODEL.CYCLE_ACR.FUSION)

        x_after = x_after.view(x_after.size(0), -1)

        x_after = F.relu(self.fc1(x_after))
        x_after = F.relu(self.fc2(x_after))

        return x_after, actor_features, None

    def fusion(self, x, out=None, type="add"):
        if out is None:
            return x
        if type == "add":
            return x + out
        elif type == "concat":
            return torch.cat([x, out], dim=1)
        else:
            raise NotImplementedError

def make_roi_action_feature_extractor(cfg, dim_in):
    func = registry.ROI_ACTION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_ACTION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, dim_in)
