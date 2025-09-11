"""
Miscellaneous utility functions
"""

import torch
import torch.nn as nn
from alphaction.structures.bounding_box import BoxList


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def pad_sequence(sequence, targ_size, padding_value=0):
    tensor_size = sequence[0].size()
    trailing_dims = tensor_size[1:]
    out_dims = (len(sequence), targ_size) + trailing_dims

    out_tensor = sequence[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequence):
        length = tensor.size(0)
        out_tensor[i, :length, ...] = tensor

    return out_tensor

# def prepare_pooled_feature(x_pooled, boxes, detach=True):
#     image_shapes = [box.size for box in boxes]
#     boxes_per_image = [len(box) for box in boxes]
#     box_tensors = [a.bbox for a in boxes]

#     if detach:
#         x_pooled = x_pooled.detach()
#     pooled_feature = x_pooled.split(boxes_per_image, dim=0)

#     boxes_result = []
#     for feature_per_image, boxes_per_image, image_shape in zip(
#             pooled_feature, box_tensors, image_shapes
#     ):
#         boxlist = BoxList(boxes_per_image, image_shape, mode="xyxy")
#         boxlist.add_field("pooled_feature", feature_per_image)
#         boxes_result.append(boxlist)
#     return boxes_result

def prepare_pooled_feature(x_pooled, boxes, is_training=False, detach=True):
    image_shapes = [box.size for box in boxes]
    boxes_per_image = [len(box) for box in boxes]
    box_tensors = [a.bbox for a in boxes]

    if detach:
        x_pooled = x_pooled.detach()
    pooled_feature = x_pooled.split(boxes_per_image, dim=0)

    boxes_result = []
    if is_training:
        for feature_per_image, boxes_per_image, image_shape in zip(
                pooled_feature, box_tensors, image_shapes
        ):
            boxlist = BoxList(boxes_per_image, image_shape, mode="xyxy")
            boxlist.add_field("pooled_feature", feature_per_image)
            boxes_result.append(boxlist)
    else:
        box_score_tensors = [box.get_field("scores") for box in boxes]
        for feature_per_image, boxes_per_image, image_shape, box_score_per_image in zip(
                pooled_feature, box_tensors, image_shapes, box_score_tensors
        ):
            boxlist = BoxList(boxes_per_image, image_shape, mode="xyxy")
            boxlist.add_field("pooled_feature", feature_per_image)
            boxlist.add_field("scores", box_score_per_image)
            boxes_result.append(boxlist)

    return boxes_result


def separate_roi_per_person(proposals, things, other_proposals, max_things):
    """
    :param things: [n2, c, t, h, w]
    :param proposals:
    :param max_things:
    :return [n, max_other, c, t, h, w]
    """
    res = []
    _, c, t, h, w = things.size()
    device = things.device
    index = 0
    for i, (person_box, other_box) in enumerate(zip(proposals, other_proposals)):
        person_num = len(person_box)
        other_num = len(other_box)
        tmp = torch.zeros((person_num, max_things, c, t, h, w), device=device)
        if other_num > max_things:
            idx = torch.randperm(other_num)[:max_things]
            tmp[:, :max_things] = things[index:index + other_num][idx]
        else:
            tmp[:, :other_num] = things[index:index + other_num]

        res.append(tmp)
        index += other_num
    features = torch.cat(res, dim=0)
    return features


def separate_batch_per_person(proposals, things):
    """
    :param things: [b, max_others, c, t, h, w]
    :return [n, max_others, c, t, h, w]
    """
    res = []
    b, max_others, c, t, h, w = things.size()
    device = things.device
    for i, person_box in enumerate(proposals):
        person_num = len(person_box)
        tmp = torch.zeros((person_num, max_others, c, t, h, w), device=device)
        tmp[:] = things[i]
        res.append(tmp)
    return torch.cat(res, dim=0)


def fuse_batch_num(things):
    n, number, c, t, h, w = things.size()
    return things.contiguous().view(-1, c, t, h, w)


def unfuse_batch_num(things, batch_size, num):
    assert things.size(0) == batch_size * num, "dimension should matches"
    _, c, t, h, w = things.size()
    return things.contiguous().view(batch_size, num, c, t, h, w)


def init_layer(layer, init_std, bias, init_scale=0.001):
    if init_std == 0:
        nn.init.constant_(layer.weight, 0)
    else:
        nn.init.normal_(layer.weight, std=init_std)
    if bias:
        nn.init.constant_(layer.bias, 0)

    # layer.weight.data.mul_(init_scale)
    # layer.bias.data.mul_(init_scale)