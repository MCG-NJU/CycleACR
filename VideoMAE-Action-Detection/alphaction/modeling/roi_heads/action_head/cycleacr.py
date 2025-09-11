from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import torch
import torch.nn as nn
from alphaction.modeling.utils import pad_sequence, prepare_pooled_feature, \
    init_layer, fuse_batch_num, separate_roi_per_person, separate_batch_per_person


class InteractionBlock(nn.Module):
    def __init__(self, dim_person, dim_other, dim_out, dim_inner, structure_config,
                 max_others=20,
                 temp_pos_len=-1,  # configs for memory feature
                 dropout=0.):
        super(InteractionBlock, self).__init__()
        self.dim_person = dim_person
        self.dim_other = dim_other
        self.dim_out = dim_out
        self.dim_inner = dim_inner
        self.max_others = max_others
        self.scale_value = dim_inner ** (-0.5)
        # config for temporal position, only used for memory feature,
        self.temp_pos_len = temp_pos_len

        bias = not structure_config.NO_BIAS
        init_std = structure_config.CONV_INIT_STD

        self.query = nn.Conv3d(dim_person, dim_inner, 1, bias)
        init_layer(self.query, init_std, bias)

        self.key = nn.Conv3d(dim_other, dim_inner, 1, bias)
        init_layer(self.key, init_std, bias)

        self.value = nn.Conv3d(dim_other, dim_inner, 1, bias)
        init_layer(self.value, init_std, bias)

        self.out = nn.Conv3d(dim_inner, dim_out, 1, bias)
        if structure_config.USE_ZERO_INIT_CONV:
            out_init = 0
        else:
            out_init = init_std
        init_layer(self.out, out_init, bias)

        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.use_ln = structure_config.LAYER_NORM

        if dim_person != dim_out:
            self.shortcut = nn.Conv3d(dim_person, dim_out, 1, bias)
            init_layer(self.shortcut, init_std, bias)
        else:
            self.shortcut = None

        if self.temp_pos_len > 0:
            self.temporal_position_k = nn.Parameter(torch.zeros(temp_pos_len, 1, self.dim_inner, 1, 1, 1))
            self.temporal_position_v = nn.Parameter(torch.zeros(temp_pos_len, 1, self.dim_inner, 1, 1, 1))

    def forward(self, person, others):
        """
        :param person: [n, channels, t, h, w]
        :param others: [n, num_other, channels, t, h, w]
        """
        device = person.device
        n, dim_person, t, h, w = person.size()
        _, max_others, dim_others, t_others, h_others, w_others = others.size()

        query_batch = person
        key = fuse_batch_num(others)  # [n*num_other, channels, t, h, w]

        query_batch = self.query(query_batch)
        key_batch = self.key(key).contiguous().view(n, self.max_others, self.dim_inner, t_others, h_others, w_others)
        value_batch = self.value(key).contiguous().view(n, self.max_others, self.dim_inner, t_others, h_others,
                                                        w_others)

        if self.temp_pos_len > 0:
            max_person_per_sec = max_others // self.temp_pos_len
            key_batch = key_batch.contiguous().view(n, self.temp_pos_len, max_person_per_sec, self.dim_inner, t_others,
                                                    h_others,
                                                    w_others)
            key_batch = key_batch + self.temporal_position_k
            key_batch = key_batch.contiguous().view(n, self.max_others, self.dim_inner, t_others,
                                                    h_others, w_others)

            value_batch = value_batch.contiguous().view(n, self.temp_pos_len, max_person_per_sec, self.dim_inner,
                                                        t_others,
                                                        h_others, w_others)
            value_batch = value_batch + self.temporal_position_v
            value_batch = value_batch.contiguous().view(n, self.max_others, self.dim_inner, t_others,
                                                        h_others, w_others)

        query_batch = query_batch.contiguous().view(n, self.dim_inner, -1).transpose(1, 2)  # [n, thw, dim_inner]
        key_batch = key_batch.contiguous().view(n, self.max_others, self.dim_inner, -1).transpose(2, 3)
        key_batch = key_batch.contiguous().view(n, self.max_others * t_others * h_others * w_others, -1).transpose(
            1, 2)

        qk = torch.bmm(query_batch, key_batch)  # n, thw, max_other * thw

        qk_sc = qk * self.scale_value

        weight = self.softmax(qk_sc)

        value_batch = value_batch.contiguous().view(n, self.max_others, self.dim_inner, -1).transpose(2, 3)
        value_batch = value_batch.contiguous().view(n, self.max_others * t_others * h_others * w_others, -1)
        out = torch.bmm(weight, value_batch)  # n, thw, dim_inner

        out = out.contiguous().view(n, t * h * w, -1)
        out = out.transpose(1, 2)
        out = out.contiguous().view(n, self.dim_inner, t, h, w)

        if self.use_ln:
            if not hasattr(self, "layer_norm"):
                self.layer_norm = nn.LayerNorm([self.dim_inner, t, h, w], elementwise_affine=False).to(
                    device)
            out = self.layer_norm(out)

        out = self.relu(out)

        out = self.out(out)
        out = self.dropout(out)

        if self.shortcut:
            person = self.shortcut(person)

        out = out + person
        return out


class CycleModeling(nn.Module):
    def __init__(self, module_cfg):
        super(CycleModeling, self).__init__()
        depth = module_cfg.DEPTH
        dropout = module_cfg.DROPOUT
        dim_inner = module_cfg.DIM_INNER
        dim_person = dim_other = dim_inner
        local_feature_len = module_cfg.LOCAL_FEATURE_LEN
        temp_feature_len = module_cfg.TEMP_FEATURE_LEN
        use_temporal_position = module_cfg.TEMPORAL_POSITION
        self.use_temporal_position = use_temporal_position

        # local branch
        local_a2c_blocks = list()
        for _ in range(depth):
            local_a2c_blocks.append(
                InteractionBlock(
                    dim_person, dim_other, dim_inner, dim_inner,
                    module_cfg, local_feature_len,
                    dropout=dropout
                )
            )
        self.local_a2c_blocks = nn.ModuleList(local_a2c_blocks)
        local_c2a_blocks = list()
        for _ in range(depth):
            local_c2a_blocks.append(
                InteractionBlock(
                    dim_person, dim_other, dim_inner, dim_inner,
                    module_cfg, temp_feature_len,
                    dropout=dropout
                )
            )
        self.local_c2a_blocks = nn.ModuleList(local_c2a_blocks)
        if use_temporal_position:
            self.local_temporal_position = nn.Parameter(torch.zeros(temp_feature_len, dim_inner, 1, 1, 1))

        # global branch
        global_a2c_blocks = list()
        for _ in range(depth):
            global_a2c_blocks.append(
                InteractionBlock(
                    dim_person, dim_other, dim_inner, dim_inner,
                    module_cfg, local_feature_len,
                    dropout=dropout
                )
            )
        self.global_a2c_blocks = nn.ModuleList(global_a2c_blocks)

    def forward(self, actor_features, context_features, local_features):
        """
        :param actor_features: [n, dim_inner, 1, 1, 1]
        :param context_features: [n, dim_inner, t_fast, 1, 1]
        :param local_features: [n, 49, dim_inner, 1, 1, 1]
        """
        global_features = context_features.mean(dim=2, keepdim=True)  # [n, dim_inner, 1, 1, 1]

        # local branch
        local_a2c_key = torch.cat([actor_features.unsqueeze(1), local_features], dim=1)  # [n, 50, dim_inner, 1, 1, 1]
        for blk in self.local_a2c_blocks:
            context_features = blk(context_features, local_a2c_key)

        local_c2a_key = context_features.unsqueeze(3).transpose(1, 2)  # [n, t_fast, dim_inner, 1, 1, 1]
        if self.use_temporal_position:
            local_c2a_key = local_c2a_key + self.local_temporal_position
        for blk in self.local_c2a_blocks:
            actor_features = blk(actor_features, local_c2a_key)

        # global branch
        global_a2c_key = torch.cat([actor_features.unsqueeze(1), local_features], dim=1)  # [n, 50, dim_inner, 1, 1, 1]
        for blk in self.global_a2c_blocks:
            global_features = blk(global_features, global_a2c_key)

        # combine outputs of two branches
        actor_features = torch.cat([actor_features, global_features], dim=1)  # [n, dim_inner*2, 1, 1, 1]
        return actor_features


class InstanceInteraction(nn.Module):
    def __init__(self, module_cfg):
        super(InstanceInteraction, self).__init__()
        depth = module_cfg.INSTANCE_DEPTH
        dropout = module_cfg.DROPOUT
        dim_inner = module_cfg.DIM_INNER
        dim_person = dim_other = dim_inner
        dim_out = module_cfg.DIM_OUT
        max_person = module_cfg.MAX_PERSON
        mem_len = module_cfg.LENGTH[0] + module_cfg.LENGTH[1] + 1
        mem_feature_len = mem_len * module_cfg.MAX_PER_SEC

        p_blocks = list()
        for _ in range(depth):
            p_blocks.append(
                InteractionBlock(
                    dim_person, dim_other, dim_inner, dim_inner,
                    module_cfg, max_person,
                    dropout=dropout
                )
            )
        self.p_blocks = nn.ModuleList(p_blocks)
        m_blocks = list()
        for _ in range(depth - 1):
            m_blocks.append(
                InteractionBlock(
                    dim_person, dim_other, dim_inner, dim_inner,
                    module_cfg, mem_feature_len,
                    temp_pos_len=mem_len if module_cfg.TEMPORAL_POSITION else -1, dropout=dropout
                )
            )
        m_blocks.append(
            InteractionBlock(
                dim_person, dim_other, dim_out, dim_inner,
                module_cfg, mem_feature_len,
                temp_pos_len=mem_len if module_cfg.TEMPORAL_POSITION else -1, dropout=dropout
            )
        )
        self.m_blocks = nn.ModuleList(m_blocks)

        self.depth = depth
        self.dim_inner = dim_inner
        self.max_person = max_person
        self.mem_len = mem_len
        self.mem_feature_len = mem_feature_len
        self.module_cfg = module_cfg

    def get_memory_feature(self, feature_pool, extras, mem_len, mem_rate, max_boxes, fixed_dim, current_x, current_box, use_penalty):
        before, after = mem_len
        mem_feature_list = []
        mem_pos_list = []
        device = current_x.device
        if use_penalty and self.training:
            cur_loss = extras["cur_loss"]
        else:
            cur_loss = 0.0
        current_feat = prepare_pooled_feature(current_x, current_box, self.training, detach=True)
        for movie_id, timestamp, new_feat in zip(extras["movie_ids"], extras["timestamps"], current_feat):
            before_inds = range(timestamp - before * mem_rate, timestamp, mem_rate)
            after_inds = range(timestamp + mem_rate, timestamp + (after + 1) * mem_rate, mem_rate)
            cache_cur_mov = feature_pool[movie_id]
            mem_box_list_before = [
                self.check_fetch_mem_feature(cache_cur_mov, mem_ind, max_boxes, cur_loss, use_penalty)
                for mem_ind in before_inds]
            mem_box_list_after = [self.check_fetch_mem_feature(cache_cur_mov, mem_ind, max_boxes, cur_loss, use_penalty)
                                  for mem_ind in after_inds]
            mem_box_current = [self.sample_mem_feature(new_feat, max_boxes), ]
            mem_box_list = mem_box_list_before + mem_box_current + mem_box_list_after
            mem_feature_list += [box_list.get_field("pooled_feature")
                                 if box_list is not None
                                 else torch.zeros(0, fixed_dim, 1, 1, 1, dtype=torch.float32, device="cuda")
                                 for box_list in mem_box_list]
            mem_pos_list += [box_list.bbox
                             if box_list is not None
                             else torch.zeros(0, 4, dtype=torch.float32, device="cuda")
                             for box_list in mem_box_list]

        seq_length = sum(mem_len) + 1
        person_per_seq = seq_length * max_boxes
        mem_feature = pad_sequence(mem_feature_list, max_boxes)
        mem_feature = mem_feature.view(-1, person_per_seq, fixed_dim, 1, 1, 1)
        mem_feature = mem_feature.to(device)
        mem_pos = pad_sequence(mem_pos_list, max_boxes)
        mem_pos = mem_pos.view(-1, person_per_seq, 4)
        mem_pos = mem_pos.to(device)

        return mem_feature, mem_pos

    def check_fetch_mem_feature(self, movie_cache, mem_ind, max_num, cur_loss, use_penalty):
        if mem_ind not in movie_cache:
            return None
        box_list = movie_cache[mem_ind]
        box_list = self.sample_mem_feature(box_list, max_num)
        if use_penalty and self.training:
            loss_tag = box_list.delete_field("loss_tag")
            penalty = loss_tag / cur_loss if loss_tag < cur_loss else cur_loss / loss_tag
            features = box_list.get_field("pooled_feature") * penalty
            box_list.add_field("pooled_feature", features)
        return box_list

    def sample_mem_feature(self, box_list, max_num):
        if len(box_list) > max_num:
            idx = torch.randperm(len(box_list))[:max_num]
            return box_list[idx].to("cuda")
        else:
            return box_list.to("cuda")

    def forward(self, person_features, proposals, extras):
        # prepare memory feature
        mem_len = self.module_cfg.LENGTH
        mem_rate = self.module_cfg.MEMORY_RATE
        use_penalty = self.module_cfg.PENALTY
        max_feature_len_per_sec = self.module_cfg.MAX_PER_SEC
        mem_feature = self.get_memory_feature(extras["person_pool"], extras, mem_len, mem_rate,
                                              max_feature_len_per_sec, self.dim_inner,
                                              person_features, proposals, use_penalty)[0]

        final_features = person_features
        for i in range(self.depth):
            person_key = separate_roi_per_person(proposals, final_features, proposals, self.max_person, )
            mem_key = separate_batch_per_person(proposals, mem_feature)
            final_features = self.p_blocks[i](final_features, person_key)
            final_features = self.m_blocks[i](final_features, mem_key)
        return final_features


class CycleACR(nn.Module):
    def __init__(self, dim_in, module_cfg):
        super(CycleACR, self).__init__()
        dropout = module_cfg.DROPOUT
        dim_inner = module_cfg.DIM_INNER
        self.dim_inner = dim_inner
        bias = not module_cfg.NO_BIAS
        conv_init_std = module_cfg.CONV_INIT_STD

        self.actor_dim_reduce = nn.Conv3d(dim_in, dim_inner, 1, bias)
        init_layer(self.actor_dim_reduce, conv_init_std, bias)
        self.reduce_dropout = nn.Dropout(dropout)

        self.context_dim_reduce = nn.Conv3d(dim_in, dim_inner, 1, bias)
        init_layer(self.context_dim_reduce, conv_init_std, bias)

        self.local_dim_reduce = nn.Conv3d(dim_in, dim_inner, 1, bias)
        init_layer(self.local_dim_reduce, conv_init_std, bias)

        self.roi_dim_reduce = nn.Conv3d(dim_inner * 2, dim_inner, 1)
        init_layer(self.roi_dim_reduce, conv_init_std, bias)

        self.cycle_modeling = CycleModeling(module_cfg)
        self.instance_interaction = InstanceInteraction(module_cfg)

    def _reduce_dim(self, actor_features, proposals, context_features, local_features):
        """
        :param actor_features: [n, dim_in, 1, 1, 1]
        :param context_features: [b, dim_in, t_fast, 1, 1]
        :param local_features: [n, dim_in, 1, 7, 7]
        """
        n = actor_features.shape[0]

        actor_features = self.actor_dim_reduce(actor_features)
        actor_features = self.reduce_dropout(actor_features)  # [n, dim_inner, 1, 1, 1]

        context_features = separate_batch_per_person(proposals, context_features.unsqueeze(1)).squeeze(1)
        context_features = self.context_dim_reduce(context_features)
        context_features = self.reduce_dropout(context_features)  # [n, dim_inner, t_fast, 1, 1]

        local_features = self.local_dim_reduce(local_features)
        local_features = self.reduce_dropout(local_features)
        local_features = local_features.reshape(n, self.dim_inner, -1).permute(0, 2, 1).reshape(n, -1, self.dim_inner, 1, 1, 1)  # [n, 49, dim_inner, 1, 1, 1]

        return actor_features, context_features, local_features

    def forward(self, actor_features, proposals, context_features, local_features, extras, part_forward):
        if part_forward != 1:
            actor_features, context_features, local_features = self._reduce_dim(actor_features, proposals, context_features, local_features)
            actor_features = self.cycle_modeling(actor_features, context_features, local_features)
            actor_features = self.roi_dim_reduce(actor_features)

            # put the context-enhanced actor features into the bank
            pooled_features = actor_features
            if part_forward == 0:
                return None, pooled_features

            final_features = self.instance_interaction(actor_features, proposals, extras)
            return final_features, pooled_features
        else:
            final_features = self.instance_interaction(actor_features, proposals, extras)
            return final_features, None


def make_cycleacr_module(cfg, dim_in):
    return CycleACR(dim_in, cfg.MODEL.CYCLE_ACR)
