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
import math
import yaml
from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
from model.nms.nms_wrapper import nms

import pdb

DEBUG = False

class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales, ratios):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales), 
            ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # top[0].reshape(1, 5)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

    def forward(self, input):

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)


        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs

        # 这里的input[0] = rpn_cls_prob.data 即rpn分类(bg or fg)的分数
        # 按照通道C取出RPN预测的框属于前景的分数，请注意，在18个channel中，前9个是框属于背景的概率，后9个才是属于前景的概率
        scores = input[0][:, self._num_anchors:, :, :]
        # input[1] = rpn_bbox_pred.data 即rpn进行bbox回归的delta
        # bbox_deltas的维度[N, num_anchors(9) * 4, H, W]
        bbox_deltas = input[1]
        # 这里的im_info的维度是[N, 2] [N, 0]表示图片width [N, 1]表示图片height
        im_info = input[2]
        cfg_key = input[3]

        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

        batch_size = bbox_deltas.size(0)
        # 获得特征图的height和width
        feat_height, feat_width = scores.size(2), scores.size(3)
        # 在feature map上通过meshgrid构建网格图
        # 这里将特征图上面的每一个点的x坐标、y坐标通过*feature_stride(在VGG-16中为16)映射到原图片中
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        # 注意:这里在原图中生成网格 shift_x shape: [height, width], shift_y shape: [height, width]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        # shifts.shape = [height*width, 4]
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        # 调用contiguous()时, 会强制拷贝一份tensor,而不是直接返回其引用
        shifts = shifts.contiguous().type_as(scores).float()
        # A = 9
        A = self._num_anchors
        # K = height * width
        K = shifts.size(0)
        # 将anchors的数据类型设置为scores张量的类型
        self._anchors = self._anchors.type_as(scores)
        # anchors.shape = (K, A, 4)
        # 这里十分地关键！
        # 一直纠结于如何在原图中生成anchor
        # 这里充分利用了python的广播特性, 原本的anchor(9*4)生成在以(0, 0)为中心点的原图中
        # 其实可以说是在原图中,也可以说是在feature map中,毕竟中心点(0, 0) * 16还是为(0, 0)
        # 然后将这个base_anchor通过与shifts相加, shift到了feature map对应的每一个像素点上！
        # 或许这就是为什么叫shifts叭, 妙哉妙哉
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        # 注意点:permute是对维度的转置 类似于tensorflow中的transpose
        # 而在pytorch中transpose只能运用于两个维度(两两交换) 使用多次transpose也能达到permute的效果
        # 这里先把bbox_deltas的维度转为[N, H, W, NUM_ANCHORS*4] 因为C中放的是bbox_deltas
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        # 这里将bbox_deltas的维度转为[N, H*W*NUM_ANCHORS, 4]
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)

        # Same story for the scores:
        # 这里与上面同理 但是需要注意这里的scores 只包含前景分数
        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)

        # Convert anchors into proposals via bbox transformations
        # 通过卷积的结果对anchors的位置进行初步的调整 
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)

        # 2. clip predicted boxes to image
        # proposals的维度[N, W * H * num_anchors, 4]
        proposals = clip_boxes(proposals, im_info, batch_size)
        
        scores_keep = scores
        proposals_keep = proposals
        # sort 函数对某一个维度进行排序(默认升序) 若没有给定维度 则默认对最后一个维度进行排序
        # sort的返回值是两个Tensor:
        # 第一个为排序好的结果 第二个为排序结果中的元素对应于原Tensor中的indice
        # 这里丢弃排序的结果 获得相应排名的indice
        # Attention:这里scores的维度是[N, H*W*NUM_ANCHORS*1]
        # 所以相当于对每个batch中所有的anchors的前景得分进行排序
        _, order = torch.sort(scores_keep, 1, True)

        # 创建与scores类型相同(device相同)的Tensor 维度为[N, 2000, 5]
        # 在训练时经过NMS操作最后只保留post_nms_topN(2000)个anchors
        output = scores.new(batch_size, post_nms_topN, 5).zero_()

        # 接下来对每一个batch进行操作
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            # 选取一个batch的 proposals 和 scores
            # proposals_single维度[H*W*num_anchors, 4]
            # scores_single维度[H*W*num_anchors]
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            # order_single维度[H*W*NUM_ANCHORS]
            order_single = order[i]

            # numel():Returns the total number of elements in the input tensor.
            # 如果pre_nms_topN小于一个batch中anchors的数量的话 就选取前面pre_nms_topN个排名比较高的anchors
            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]

            # 选取对应的anchors的坐标以及对应的分数
            # proposals_single维度[pre_nms_topN, 4]
            # scores_single维度[pre_nms_topN, 1]
            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)


            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)5
            # 接下来对anchors使用非极大值抑制 阈值设置为0.7
            # 详细可以参考https://zhuanlan.zhihu.com/p/54709759
            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh, force_cpu=not cfg.USE_GPU_NMS)
            # nms返回的为anchors(proposals)的 idx
            keep_idx_i = keep_idx_i.long().view(-1)

            # 这里选取post_nms_topN个(训练时为2000 测试时为300) 得分较高的proposals
            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            # 分别取出post_nms_topN个proposals的坐标和分数
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            # output的维度为[batch_size, post_nms_topN, 5]
            num_proposal = proposals_single.size(0)
            # 填写batch的信息和anchor的坐标
            # 第0维不重要 最后一维的第一个数表示batch号 后四个数为proposals的坐标
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single

        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1,1).expand_as(ws)) & (hs >= min_size.view(-1,1).expand_as(hs)))
        return keep
