# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Xizhou Zhu, Yuwen Xiong, Bin Xiao
# --------------------------------------------------------

import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()

config.MXNET_VERSION = ''
config.output_path = ''
config.symbol = ''
config.gpus = ''
config.CLASS_AGNOSTIC = True
config.SCALES = [(360, 600)]  # first is scale (the shorter side); second is max size

# default training
config.default = edict()
config.default.frequent = 20
config.default.kvstore = 'device'

# network related params
config.network = edict()
config.network.pretrained = ''
config.network.pretrained_flow = ''
config.network.pretrained_epoch = 0
config.network.PIXEL_MEANS = np.array([0, 0, 0])
config.network.IMAGE_STRIDE = 0
config.network.FIXED_PARAMS = ['gamma', 'beta']
config.network.DFF_FEAT_DIM = 2048

# dataset related params
config.dataset = edict()
config.dataset.dataset = 'CityScape'
config.dataset.image_set = 'leftImg8bit_train'
config.dataset.test_image_set = 'leftImg8bit_val'
config.dataset.root_path = '../data'
config.dataset.dataset_path = '../data/cityscapes'
config.dataset.NUM_CLASSES = 19
config.dataset.annotation_prefix = 'gtFine'


config.TRAIN = edict()
config.TRAIN.lr = 0
config.TRAIN.lr_step = ''
config.TRAIN.lr_factor = 0.1
config.TRAIN.warmup = False
config.TRAIN.warmup_lr = 0
config.TRAIN.warmup_step = 0
config.TRAIN.momentum = 0.9
config.TRAIN.wd = 0.0005
config.TRAIN.begin_epoch = 0
config.TRAIN.end_epoch = 0
config.TRAIN.model_prefix = ''

# whether resume training
config.TRAIN.RESUME = False
# whether flip image
config.TRAIN.FLIP = True
# whether shuffle image
config.TRAIN.SHUFFLE = True
# whether use OHEM
config.TRAIN.ENABLE_OHEM = False
# size of images for each device, 2 for rcnn, 1 for rpn and e2e
config.TRAIN.BATCH_IMAGES = 1
# e2e changes behavior of anchor loader and metric
config.TRAIN.END2END = False
# group images with similar aspect ratio
config.TRAIN.ASPECT_GROUPING = True

# used for end2end training

# DFF, trained image sampled from [min_offset, max_offset]
config.TRAIN.MIN_OFFSET = -4
config.TRAIN.MAX_OFFSET = 0

config.TEST = edict()
# size of images for each device
config.TEST.BATCH_IMAGES = 1

# DFF
config.TEST.KEY_FRAME_INTERVAL = 5

config.TEST.max_per_image = 300

# Test Model Epoch
config.TEST.test_epoch = 0


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    if k == 'TRAIN':
                        if 'BBOX_WEIGHTS' in v:
                            v['BBOX_WEIGHTS'] = np.array(v['BBOX_WEIGHTS'])
                    elif k == 'network':
                        if 'PIXEL_MEANS' in v:
                            v['PIXEL_MEANS'] = np.array(v['PIXEL_MEANS'])
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("key must exist in config.py")
