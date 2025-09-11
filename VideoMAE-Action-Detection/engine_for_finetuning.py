import os
import numpy as np
import math
import sys
import time
import datetime
import logging
from typing import Iterable, Optional
import torch
import torch.nn as nn

import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import utils
from alphaction.modeling.utils import cat
from alphaction.structures.bounding_box import BoxList
from data.ava_eval import do_ava_evaluation
from alphaction.utils.comm import get_rank, is_main_process, all_gather, gather, synchronize, get_world_size
from alphaction.structures.memory_pool import MemoryPool


def train_class_batch(model, samples, boxes, mem_extras):

    outputs, pooled_feature = model(samples, boxes, mem_extras)

    labels = cat([proposal.get_field("labels") for proposal in boxes], dim=0)
    assert outputs.shape[1] == labels.shape[1], \
        "The shape of tensor class logits doesn't match the label tensor."
    # loss = criterion(outputs, target)
    batch_size = outputs.shape[0]
    loss = F.binary_cross_entropy_with_logits(outputs, labels.to(dtype=torch.float32), reduction='mean')
    loss = loss * batch_size

    return loss, outputs, pooled_feature


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None,
                    mem_active=False, person_pool=MemoryPool()):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    loss_value = torch.tensor(0.0).item()

    for step, (samples, boxes, extras, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None:
        # if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        boxes = [box.to(device=device) for box in boxes]  # boxlist
        mem_extras = {}
        if mem_active:
            movie_ids = [e["movie_id"] for e in extras]
            timestamps = [e["timestamp"] for e in extras]
            mem_extras["person_pool"] = person_pool
            mem_extras["movie_ids"] = movie_ids
            mem_extras["timestamps"] = timestamps
            mem_extras["cur_loss"] = loss_value

        # targets = targets.to(device, non_blocking=True)

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            loss, _, pooled_feature = train_class_batch(
                model, samples, boxes, mem_extras)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order)
            optimizer.zero_grad()
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        # update mem pool
        if mem_active:
            person_feature = [ft.to("cpu") for ft in pooled_feature[0]]

            feature_update = MemoryPool()
            for movie_id, timestamp, new_feature in zip(movie_ids, timestamps, person_feature):
                new_feature.add_field("loss_tag", loss_value)
                feature_update[movie_id, timestamp] = new_feature

            feature_update_list = all_gather(feature_update)
            person_pool.update_list(feature_update_list)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, person_pool


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, output_dir, epoch, log_writer=None):
    if not utils.is_main_process():
        return

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    logging.info("Start evaluation on ava_v2.2 dataset({} videos).".format(data_loader.num_samples))

    cpu_device = torch.device("cpu")
    results_dict = {}

    loader_len = len(data_loader)
    person_feature_pool = MemoryPool()
    batch_info_list = [None]*loader_len
    logging.info("Stage 1: extracting clip features.")
    start_time = time.time()

    for i, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # if i > 10:
        #     break
        videos = batch[0]
        boxes = batch[1]
        extras = batch[2]
        video_ids = batch[3]

        videos = videos.to(device, non_blocking=True)
        boxes = [box.to(device=device) for box in boxes]  # boxlist
        movie_ids = [e["movie_id"] for e in extras]
        timestamps = [e["timestamp"] for e in extras]
        # target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            feature = model(videos, boxes, part_forward=0)[1]
        person_feature = [ft.to(cpu_device) for ft in feature[0]]
        roi_feature = [ft.to(cpu_device) for ft in feature[1]]

        # store person features into memory pool
        for movie_id, timestamp, p_ft, _ in zip(movie_ids, timestamps, person_feature, roi_feature):
            person_feature_pool[movie_id, timestamp] = p_ft
        # store other information in list, for further inference
        batch_info_list[i] = (movie_ids, timestamps, video_ids, roi_feature)

    # gather feature pools from different ranks
    # synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # logging.info(
    #     "Stage 1 time: {} ({} s / video per device, on {} devices)".format(
    #         total_time_str, total_time * num_devices / len(dataset), num_devices
    #     )
    # )

    all_feature_pool_p = person_feature_pool

    # do the inference
    logging.info("Stage 2: predicting with extracted feature.")
    start_time = time.time()

    for movie_ids, timestamps, video_ids, roi_feature in metric_logger.log_every(batch_info_list, 10, header):
        current_feat_p = [all_feature_pool_p[movie_id, timestamp].to(device)
                          for movie_id, timestamp in zip(movie_ids, timestamps)]
        current_feat_r = [ft_r.to(device) for ft_r in roi_feature]
        extras = dict(
            person_pool=all_feature_pool_p,
            movie_ids=movie_ids,
            timestamps=timestamps,
            current_feat_p=current_feat_p,
            current_feat_r=current_feat_r,
        )
        with torch.cuda.amp.autocast():
            output = model(None, None, extras=extras, part_forward=1)[0]
            # output = postprocess(output, boxes)
        output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {video_id: result for video_id, result in zip(video_ids, output)}
        )

    # synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # logging.info(
    #     "Stage 2 time: {} ({} s / video per device, on {} devices)".format(
    #         total_time_str, total_time * num_devices / len(dataset), num_devices
    #     )
    # )

    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=total_time))
    # logging.info(
    #     "Total inference time: {}".format(total_time_str)
    # )

    # convert a dict where the key is the index in a list
    video_ids = list(sorted(results_dict.keys()))
    if len(video_ids) != video_ids[-1] + 1:
        logging.warning(
            "Number of videos that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )
    # convert to a list
    predictions = [results_dict[i] for i in video_ids]
    print(f"length of predictions: {len(predictions)}")

    logging.info("Performing ava evaluation")

    output_folder = os.path.join(output_dir, "inference")
    os.makedirs(output_folder, exist_ok=True)

    eval_res = do_ava_evaluation(
        dataset=data_loader.dataset,
        predictions=predictions,
        output_folder=output_folder,
    )

    if log_writer is not None:
        eval_res, _ = eval_res
        total_mAP = eval_res['PascalBoxes_Precision/mAP@0.5IOU']
        log_writer.update(map=total_mAP, head="perf", step=epoch)

