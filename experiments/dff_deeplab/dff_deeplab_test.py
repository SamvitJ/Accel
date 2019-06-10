# --------------------------------------------------------
# Accel
# Copyright (c) 2019
# Licensed under the MIT License [see LICENSE for details]
# Modified by Samvit Jain
# --------------------------------------------------------

import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'dff_rfcn'))

import test

if __name__ == "__main__":
    test.main()
