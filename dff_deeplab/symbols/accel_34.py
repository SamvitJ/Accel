# --------------------------------------------------------
# Accel
# Copyright (c) 2019
# Licensed under the MIT License [see LICENSE for details]
# Modified by Samvit Jain
# --------------------------------------------------------

import cPickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_target import *
from operator_py.box_annotator_ohem import *
from operator_py.rpn_inv_normalize import *
from operator_py.tile_as import *

from resnet_v1_101_flownet_deeplab import *

class accel_34(resnet_v1_101_flownet_deeplab):

    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        self.units = (3, 4, 23, 3) # use for 101
        self.filter_list = [256, 512, 1024, 2048]

    def get_train_symbol(self, cfg):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_interms = cfg.TRAIN.KEY_INTERVAL - 1

        data = mx.sym.Variable(name="data")
        data_ref = mx.sym.Variable(name="data_ref")
        eq_flag = mx.sym.Variable(name="eq_flag")
        seg_cls_gt = mx.symbol.Variable(name='label')

        # keyframe features
        data_ref_split = mx.sym.split(data_ref, num_outputs=num_interms, axis=0)
        conv_feat = self.get_resnet_dcn(data_ref_split[0])

        # data
        data_next = mx.sym.Concat(*[data_ref_split[1], data_ref_split[2], data_ref_split[3], data], dim=0)
        data_prev = mx.sym.Concat(*[data_ref_split[0], data_ref_split[1], data_ref_split[2], data_ref_split[3]], dim=0)

        # warp features
        flow, scale_map = self.get_flownet(data_next, data_prev)
        flow_grid = mx.sym.GridGenerator(data=flow, transform_type='warp', name='flow_grid')
        flow_grid_split = mx.sym.split(flow_grid, num_outputs=num_interms, axis=0)

        for idx in range(num_interms):
            conv_feat = mx.sym.BilinearSampler(data=conv_feat, grid=flow_grid_split[idx], name='warping_feat')

        # L branch
        fc6_bias = mx.symbol.Variable('fc6_bias', lr_mult=2.0)
        fc6_weight = mx.symbol.Variable('fc6_weight', lr_mult=1.0)

        fc6 = mx.symbol.Convolution(data=conv_feat, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="fc6",
                                    bias=fc6_bias, weight=fc6_weight, workspace=self.workspace)
        relu_fc6 = mx.sym.Activation(data=fc6, act_type='relu', name='relu_fc6')

        score_bias = mx.symbol.Variable('score_bias', lr_mult=2.0)
        score_weight = mx.symbol.Variable('score_weight', lr_mult=1.0)

        score = mx.symbol.Convolution(data=relu_fc6, kernel=(1, 1), pad=(0, 0), num_filter=num_classes, name="score",
                                      bias=score_bias, weight=score_weight, workspace=self.workspace)

        upsampling = mx.symbol.Deconvolution(data=score, num_filter=num_classes, kernel=(32, 32), stride=(16, 16),
                                             num_group=num_classes, no_bias=True, name='upsampling',
                                             attr={'lr_mult': '0.0'}, workspace=self.workspace)

        croped_score = mx.symbol.Crop(*[upsampling, data], offset=(8, 8), name='croped_score')

        # R branch
        feat_curr = self.resnet(data_sym=data, prefix="34_", units=[3, 4, 6], num_stages=3,
            filter_list=[64, 64, 128, 256, 512], num_classes=1000, data_type="imagenet",
            bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)
        feat_curr = self.get_resnet_dcn_34_conv5(feat_curr)
        feat_curr = mx.symbol.Deconvolution(data=feat_curr, num_filter=2048, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
            no_bias=True, name='34_feat_upsampling', workspace=self.workspace,
            attr={'lr_mult': '2.0'})

        curr_fc6_bias = mx.symbol.Variable('34_fc6_bias', lr_mult=2.0)
        curr_fc6_weight = mx.symbol.Variable('34_fc6_weight', lr_mult=1.0)

        curr_fc6 = mx.symbol.Convolution(data=feat_curr, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="34_fc6",
                                         bias=curr_fc6_bias, weight=curr_fc6_weight, workspace=self.workspace)
        curr_relu_fc6 = mx.sym.Activation(data=curr_fc6, act_type='relu', name='34_relu_fc6')

        curr_score_bias = mx.symbol.Variable('34_score_bias', lr_mult=2.0)
        curr_score_weight = mx.symbol.Variable('34_score_weight', lr_mult=1.0)

        curr_score = mx.symbol.Convolution(data=curr_relu_fc6, kernel=(1, 1), pad=(0, 0), num_filter=num_classes, name="34_score",
                                           bias=curr_score_bias, weight=curr_score_weight, workspace=self.workspace)

        curr_upsampling = mx.symbol.Deconvolution(data=curr_score, num_filter=num_classes, kernel=(32, 32), stride=(16, 16),
                                                  num_group=num_classes, no_bias=True, name='34_upsampling',
                                                  attr={'lr_mult': '0.0'}, workspace=self.workspace)

        curr_croped_score = mx.symbol.Crop(*[curr_upsampling, data], offset=(8, 8), name='34_croped_score')

        # correction layer
        stacked_in = mx.sym.Concat(*[croped_score, curr_croped_score], dim=1)

        corr_bias = mx.symbol.Variable('corr_bias', lr_mult=4.0)
        corr_weight = mx.symbol.Variable('corr_weight', lr_mult=2.0)
        correction = mx.symbol.Convolution(data=stacked_in, kernel=(1, 1), pad=(0, 0), num_filter=num_classes, name="correction",
                                           bias=corr_bias, weight=corr_weight, workspace=self.workspace)

        softmax = mx.symbol.SoftmaxOutput(data=correction, label=seg_cls_gt, normalization='valid', multi_output=True,
                                          use_ignore=True, ignore_label=255, name="softmax")

        group = mx.sym.Group([softmax, data_ref, eq_flag])
        self.sym = group
        return group

    def get_key_test_symbol(self, cfg):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        data = mx.sym.Variable(name="data")
        data_key = mx.sym.Variable(name="data_key")
        feat_key = mx.sym.Variable(name="feat_key")

        # shared convolutional layers
        conv_feat = self.get_resnet_dcn(data)

        # deeplab
        fc6_bias = mx.symbol.Variable('fc6_bias', lr_mult=2.0)
        fc6_weight = mx.symbol.Variable('fc6_weight', lr_mult=1.0)

        fc6 = mx.symbol.Convolution(
            data=conv_feat, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="fc6", bias=fc6_bias, weight=fc6_weight,
            workspace=self.workspace)
        relu_fc6 = mx.sym.Activation(data=fc6, act_type='relu', name='relu_fc6')

        score_bias = mx.symbol.Variable('score_bias', lr_mult=2.0)
        score_weight = mx.symbol.Variable('score_weight', lr_mult=1.0)

        score = mx.symbol.Convolution(
            data=relu_fc6, kernel=(1, 1), pad=(0, 0), num_filter=num_classes, name="score", bias=score_bias,
            weight=score_weight, workspace=self.workspace)

        upsampling = mx.symbol.Deconvolution(
            data=score, num_filter=num_classes, kernel=(32, 32), stride=(16, 16), num_group=num_classes, no_bias=True,
            name='upsampling', attr={'lr_mult': '0.0'}, workspace=self.workspace)

        croped_score = mx.symbol.Crop(*[upsampling, data], offset=(8, 8), name='croped_score')

        group = mx.sym.Group([data_key, feat_key, conv_feat, croped_score])
        self.sym = group
        return group

    def get_cur_test_symbol(self, cfg):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        data_cur = mx.sym.Variable(name="data")
        data_key = mx.sym.Variable(name="data_key")
        conv_feat = mx.sym.Variable(name="feat_key")

        # warp features
        flow, scale_map = self.get_flownet(data_cur, data_key)
        flow_grid = mx.sym.GridGenerator(data=flow, transform_type='warp', name='flow_grid')
        conv_feat = mx.sym.BilinearSampler(data=conv_feat, grid=flow_grid, name='warping_feat')

        # L branch
        fc6_bias = mx.symbol.Variable('fc6_bias', lr_mult=2.0)
        fc6_weight = mx.symbol.Variable('fc6_weight', lr_mult=1.0)

        fc6 = mx.symbol.Convolution(
            data=conv_feat, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="fc6", bias=fc6_bias, weight=fc6_weight,
            workspace=self.workspace)
        relu_fc6 = mx.sym.Activation(data=fc6, act_type='relu', name='relu_fc6')

        score_bias = mx.symbol.Variable('score_bias', lr_mult=2.0)
        score_weight = mx.symbol.Variable('score_weight', lr_mult=1.0)

        score = mx.symbol.Convolution(
            data=relu_fc6, kernel=(1, 1), pad=(0, 0), num_filter=num_classes, name="score", bias=score_bias,
            weight=score_weight, workspace=self.workspace)

        upsampling = mx.symbol.Deconvolution(
            data=score, num_filter=num_classes, kernel=(32, 32), stride=(16, 16), num_group=num_classes, no_bias=True,
            name='upsampling', attr={'lr_mult': '0.0'}, workspace=self.workspace)

        croped_score = mx.symbol.Crop(*[upsampling, data_cur], offset=(8, 8), name='croped_score')

        # R branch
        feat_curr = self.resnet(data_sym=data_cur, prefix="34_", units=[3, 4, 6], num_stages=3,
            filter_list=[64, 64, 128, 256, 512], num_classes=1000, data_type="imagenet",
            bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)
        feat_curr = self.get_resnet_dcn_34_conv5(feat_curr)
        feat_curr = mx.symbol.Deconvolution(data=feat_curr, num_filter=2048, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
            no_bias=True, name='34_feat_upsampling', workspace=self.workspace,
            attr={'lr_mult': '2.0'})

        curr_fc6_bias = mx.symbol.Variable('34_fc6_bias', lr_mult=2.0)
        curr_fc6_weight = mx.symbol.Variable('34_fc6_weight', lr_mult=1.0)

        curr_fc6 = mx.symbol.Convolution(
            data=feat_curr, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="34_fc6", bias=curr_fc6_bias,
            weight=curr_fc6_weight, workspace=self.workspace)
        curr_relu_fc6 = mx.sym.Activation(data=curr_fc6, act_type='relu', name='34_relu_fc6')

        curr_score_bias = mx.symbol.Variable('34_score_bias', lr_mult=2.0)
        curr_score_weight = mx.symbol.Variable('34_score_weight', lr_mult=1.0)

        curr_score = mx.symbol.Convolution(
            data=curr_relu_fc6, kernel=(1, 1), pad=(0, 0), num_filter=num_classes, name="34_score",
            bias=curr_score_bias, weight=curr_score_weight, workspace=self.workspace)

        curr_upsampling = mx.symbol.Deconvolution(
            data=curr_score, num_filter=num_classes, kernel=(32, 32), stride=(16, 16), num_group=num_classes,
            no_bias=True, name='34_upsampling', attr={'lr_mult': '0.0'}, workspace=self.workspace)

        curr_croped_score = mx.symbol.Crop(*[curr_upsampling, data_cur], offset=(8, 8), name='34_croped_score')

        # correction layer
        stacked_in = mx.sym.Concat(*[croped_score, curr_croped_score], dim=1)

        corr_bias = mx.symbol.Variable('corr_bias', lr_mult=4.0)
        corr_weight = mx.symbol.Variable('corr_weight', lr_mult=2.0)
        correction = mx.symbol.Convolution(data=stacked_in, kernel=(1, 1), pad=(0, 0), num_filter=num_classes, name="correction",
                                           bias=corr_bias, weight=corr_weight, workspace=self.workspace)

        group = mx.sym.Group([data_key, conv_feat, correction])
        self.sym = group
        return group

    def get_batch_test_symbol(self, cfg):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        data_key = mx.sym.Variable(name="data_key")
        data_other = mx.sym.Variable(name="data_other")
        im_info = mx.sym.Variable(name="im_info")

        # shared convolutional layers
        conv_feat_key = self.get_resnet_v1(data_key)

        data_key_tiled = mx.sym.Custom(data_content=data_key, data_shape=data_other, op_type='tile_as')
        conv_feat_key_tiled = mx.sym.Custom(data_content=conv_feat_key, data_shape=data_other, op_type='tile_as')
        flow, scale_map = self.get_flownet(data_other, data_key_tiled)
        flow_grid = mx.sym.GridGenerator(data=flow, transform_type='warp', name='flow_grid')
        conv_feat_other = mx.sym.BilinearSampler(data=conv_feat_key_tiled, grid=flow_grid, name='warping_feat')
        conv_feat_other = conv_feat_other * scale_map

        conv_feat = mx.symbol.Concat(conv_feat_key, conv_feat_other, dim=0)

        conv_feats = mx.sym.SliceChannel(conv_feat, axis=1, num_outputs=2)

        # RPN
        rpn_feat = conv_feats[0]
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

        if cfg.network.NORMALIZE_RPN:
            rpn_bbox_pred = mx.sym.Custom(
                bbox_pred=rpn_bbox_pred, op_type='rpn_inv_normalize', num_anchors=num_anchors,
                bbox_mean=cfg.network.ANCHOR_MEANS, bbox_std=cfg.network.ANCHOR_STDS)

        # ROI Proposal
        rpn_cls_score_reshape = mx.sym.Reshape(
            data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
        rpn_cls_prob = mx.sym.SoftmaxActivation(
            data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
        rpn_cls_prob_reshape = mx.sym.Reshape(
            data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
        if cfg.TEST.CXX_PROPOSAL:
            rois = mx.contrib.sym.MultiProposal(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
        else:
            NotImplemented

        # res5
        rfcn_feat = conv_feats[1]
        rfcn_cls = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7*7*num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7*7*4*num_reg_classes, name="rfcn_bbox")
        psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, group_size=7, pooled_size=7,
                                                   output_dim=num_classes, spatial_scale=0.0625)
        psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, group_size=7, pooled_size=7,
                                                   output_dim=8, spatial_scale=0.0625)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(7, 7))

        # classification
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
        # bounding box regression
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))

        # reshape output
        cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
        bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_pred_reshape')

        # group output
        group = mx.sym.Group([rois, cls_prob, bbox_pred])
        self.sym = group
        return group

    def init_weight(self, cfg, arg_params, aux_params):
        arg_params['corr_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['corr_weight'])
        arg_params['corr_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['corr_bias'])

