import torch

from .roi_action_feature_extractor import make_roi_action_feature_extractor
from .roi_action_predictors import make_roi_action_predictor
from .inference import make_roi_action_post_processor
from .loss import make_roi_action_loss_evaluator
from .metric import make_roi_action_accuracy_evaluator
from alphaction.modeling.utils import prepare_pooled_feature, cat
from alphaction.utils.comm import all_reduce


class ROIActionHead(torch.nn.Module):
    """
    Generic Action Head class.
    """

    def __init__(self, cfg, dim_in):
        super(ROIActionHead, self).__init__()
        self.feature_extractor = make_roi_action_feature_extractor(cfg, dim_in)
        self.predictor = make_roi_action_predictor(cfg, self.feature_extractor.dim_out)
        self.post_processor = make_roi_action_post_processor(cfg)
        # self.loss_evaluator = make_roi_action_loss_evaluator(cfg)
        # self.accuracy_evaluator = make_roi_action_accuracy_evaluator(cfg)
        self.proposal_per_clip = cfg.MODEL.ROI_ACTION_HEAD.PROPOSAL_PER_CLIP
        self.test_ext = cfg.TEST.EXTEND_SCALE

    def sample_box(self, boxes):
        proposals = []
        num_proposals = self.proposal_per_clip
        for boxes_per_image in boxes:
            num_boxes = len(boxes_per_image)

            if num_boxes > num_proposals:
                choice_inds = torch.randperm(num_boxes)[:num_proposals]
                proposals_per_image = boxes_per_image[choice_inds]
            else:
                proposals_per_image = boxes_per_image
            proposals_per_image = proposals_per_image.random_aug(0.2, 0.1, 0.1, 0.05)
            proposals.append(proposals_per_image)
        return proposals

    def forward(self, slow_features, fast_features, boxes, extras={}, part_forward=-1):
        # In training stage, boxes are from gt.
        # In testing stage, boxes are detected by human detector and proposals should be
        # enlarged boxes.
        assert not (self.training and part_forward >= 0)

        if part_forward == 1:
            boxes = extras["current_feat_p"]

        if self.training:
            proposals = self.sample_box(boxes)
        else:
            proposals = [box.extend(self.test_ext) for box in boxes]

        x, actor_features, roi_features = self.feature_extractor(slow_features, fast_features, proposals, extras, part_forward)

        if part_forward == 0:
            actor_pooled_feature = prepare_pooled_feature(actor_features, boxes)
            roi_pooled_feature = prepare_pooled_feature(roi_features, boxes)
            return None, [actor_pooled_feature, roi_pooled_feature]

        action_logits = self.predictor(x)

        if not self.training:
            result = self.post_processor(action_logits, boxes)
            return result, None

        actor_pooled_feature = prepare_pooled_feature(actor_features, proposals, self.training)

        return (
            action_logits,
            [actor_pooled_feature, ],
        )

    def c2_weight_mapping(self):
        weight_map = {}
        for name, m_child in self.named_children():
            if m_child.state_dict() and hasattr(m_child, "c2_weight_mapping"):
                child_map = m_child.c2_weight_mapping()
                for key, val in child_map.items():
                    new_key = name + '.' + key
                    weight_map[new_key] = val
        return weight_map


def build_roi_action_head(cfg, dim_in):
    return ROIActionHead(cfg, dim_in)
