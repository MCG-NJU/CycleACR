import torch
from torch import nn
import torch.nn.functional as F

from alphaction.structures.bounding_box import BoxList
from alphaction.modeling.utils import cat


class PostProcessor(nn.Module):

    def forward(self, class_logits, boxes):
        # boxes should be (#detections,4)
        # prob should be calculated in different way.
        class_logits = torch.sigmoid(class_logits)
        box_scores = cat([box.get_field("scores") for box in boxes], dim=0)
        box_scores = box_scores.reshape(class_logits.shape[0], 1)
        action_prob = class_logits * box_scores

        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        box_tensors = [a.bbox for a in boxes]

        action_prob = action_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_image, image_shape in zip(
                action_prob, box_tensors, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_image, prob, image_shape)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist


def make_roi_action_post_processor(cfg):
    postprocessor = PostProcessor()
    return postprocessor
