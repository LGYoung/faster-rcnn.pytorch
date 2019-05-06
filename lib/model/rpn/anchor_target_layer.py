from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_transform_batch

import pdb

DEBUG = False

try:
    long        # Python 2
except NameError:
    long = int  # Python 3

'''
    anchor_target_layer.py
    将anchor boxes(没有经过任何操作的 没有经过网络模型预测值的)与ground truth boxes对比
    对于每个anchor boxes 都计算出它和每个ground truth boxes 之间的IOU值 得到overlap
    对于每个anchor boxes 遍历所有ground truth，找到它与所有ground truth最大的overlap值
    得到   max_overlaps   shape [#anchor_boxes]
    对于每个ground truth boxes 遍历所有的anchor boxes 找到它与所有anchor boxes最大的overlap 值
    得到   gt_max_overlaps  shape [#gt_boxes]

    选择出所有正样本：
        (1)对于每个ground truth boxes 与它具有最大的overlap 的anchor boxes是正样本
        (2)对于每个anchor boxes 只要它与任意的ground truth boxes之间的IOU值大于0.7
    选择出正样本后 对所有前景正样本进行座标编码（generate good bounding boxes regression coefficients）
    实际上代码实现的时候 是对图像中的每个anchor boxes都分配了ground truth boxes值 无论最后anchor boxes被分为正样本
    还是负样本 anchor boxes与哪个gt boxes的overlap最大 就认为它是哪个gt boxes 的正样本 然后进行位置编码

    所有overlap 小于0.3的anchor记作负样本
'''

# Anchor_target_layer作用是将anchors与GT联系在一起，生成标签和bbox regression的目标
class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, scales, ratios):
        super(_AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride
        self._scales = scales
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0

    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors

        # gt_boxes维度[batch_size, 20, 5] 为什么会是20呢 请看config.py中的MAX_NUM_GT_BOXES参数
        # im_info维度[batch_size, 3]  3表示(h, w, scale)——scale是最小边resize到600的需要乘的系数
        # rpn_cls_score的维度[N, 2*9, H, W]
        rpn_cls_score = input[0]
        gt_boxes = input[1]
        im_info = input[2]
        num_boxes = input[3]

        # map of shape (..., H, W)
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)

        batch_size = gt_boxes.size(0)

        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        # 这里同proposal_layer中使用的方法一样 生成所有的anchors 保存在all_anchors中
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        # 制造二维的网格
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        # ravel()=reshape(-1)
        # 将网格 变为(x1 y1 x2 y2)形式
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()

        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(gt_boxes) # move to specific gpu.
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        # all_anchors的维度[K * A , 4] ([height * width * 9, 4])
        all_anchors = all_anchors.view(K * A, 4)

        total_anchors = int(K * A)
        # 这里的im_info的维度是[N, 2] [N, 0]表示图片width [N, 1]表示图片height
        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < long(im_info[0][1]) + self._allowed_border) &
                (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border))
        # 将那些满足keep条件的 anchors 的索引设置为1
        inds_inside = torch.nonzero(keep).view(-1)
        # 保留满足条件的索引
        # keep only inside anchors
        # anchors的维度[N, 4]
        anchors = all_anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        # labels维度[B, N]
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)
        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
        # 返回overlaps的维度[B，N，K]
        overlaps = bbox_overlaps_batch(anchors, gt_boxes)
        # 求每一个anchor 与哪个 gts 的交并比最大的那个 以及 indices
        # 返回的维度为[B, N]
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        # 求每一个 gt 与哪个anchor的交并比最大 返回最大的
        # 返回的维度为[B, K]
        gt_max_overlaps, _ = torch.max(overlaps, 1)

        # 如果不需要抑制positive的anchor 就先给背景anchor赋值 这样在赋前景值的时候可以覆盖
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # 对于那些和 所有gt的交并比 都小于阈值(最大IoU<0.3)的anchors而言 它们的label就为0
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        # 将那些与anchors没有交集的gt 设置为1e-5 防止在下面使用eq()函数的时候造成错误
        gt_max_overlaps[gt_max_overlaps==0] = 1e-5
        # 将gt_max_overlaps和overlaps比较 相等的只有可能是 gt 与 anchors IoU 最大的
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)
        # 将与 gt 有最大IoU的anchors设置为正样例
        if torch.sum(keep) > 0:
            labels[keep>0] = 1

        # fg label: above threshold IOU
        # 对于那些和 所有gt的交并比 大于阈值(IoU>0.7)的anchors而言 它们的label就为1
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
        # 这个参数就是看positive与negative谁比较强，先设置0说明positive强，因为0可能转1,而后设置0说明negative强，设置完1还可以设置成0
        # 默认为positive强 所以它默认为False
        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        # 在一个batch中 fg最大的数值(0.5 * 256)
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        # 求batch中每张图片 fg 和 bg 的总数
        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)

        
        '''
            训练RPN的batch size=256
            表示计算RPN损失函数时前景/背景或正/负样本共计算256个anchor boxes的损失值
            这个batch size并不体现在任何前向传播的操作中，只是表示RPN选择多少个样本计算损失 
        
            就是说，对于image_batch_size数量的输入图像前向传播到了RPN，
            对于同一batch size的每一张图像都会生成相同数量，相同座标的anchor boxes
            对于每一张图像就选择256个样本计算损失   
            并不是在一个batch size 的anchor boxes中一起进行选择的
        '''
        # 对batch中的每一张图片进行操作
        for i in range(batch_size):
            # subsample positive labels if we have too many
            # 如果一张图片中的正样本数大于256 * 0.5
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                # 对fd_inds进行随机排列 然后对除了前128个 的所有fg 的labels设置为-1(don't care)
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                labels[i][disable_inds] = -1

            # num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
            # 求一张图片中 bg 的最大数目
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]
            # 同理
            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1

        offset = torch.arange(0, batch_size)*gt_boxes.size(1)

        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
        bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))

        # use a single value instead of 4 values for easy index.
        bbox_inside_weights[labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]

        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(labels[i] >= 0)
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights

        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        outputs = []

        labels = labels.view(batch_size, height, width, A).permute(0,3,1,2).contiguous()
        labels = labels.view(batch_size, 1, A * height, width)
        outputs.append(labels)

        bbox_targets = bbox_targets.view(batch_size, height, width, A*4).permute(0,3,1,2).contiguous()
        outputs.append(bbox_targets)

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)

        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()

        outputs.append(bbox_inside_weights)

        bbox_outside_weights = bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        outputs.append(bbox_outside_weights)

        return outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])
