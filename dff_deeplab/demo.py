# --------------------------------------------------------
# Accel
# Copyright (c) 2019
# Licensed under the MIT License [see LICENSE for details]
# Modified by Samvit Jain
# --------------------------------------------------------

import _init_paths

import argparse
import os
import glob
import sys
import logging
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
from PIL import Image
import numpy as np

# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/dff_deeplab/cfgs/dff_deeplab_vid_demo.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from core.tester import im_segment, Predictor
from symbols import *
from utils.load_model import load_param, load_param_multi
from utils.show_boxes import show_boxes, draw_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

def parse_args():
    parser = argparse.ArgumentParser(description='Show Deep Feature Flow demo')
    parser.add_argument('-v', '--version', type=int, default=18)
    parser.add_argument('-i', '--interval', type=int, default=1)
    parser.add_argument('-e', '--num_ex', type=int, default=10)
    parser.add_argument('--avg', dest='avg_acc', action='store_true')
    parser.set_defaults(avg_acc=False)
    args = parser.parse_args()
    return args

args = parse_args()

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.true_divide(np.diag(hist), (hist.sum(1) + hist.sum(0) - np.diag(hist)))

def getpallete(num_cls):
    """
    this function is to get the colormap for visualizing the segmentation mask
    :param num_cls: the number of visulized class
    :return: the pallete
    """
    n = num_cls
    pallete_raw = np.zeros((n, 3)).astype('uint8')
    pallete = np.zeros((n, 3)).astype('uint8')

    pallete_raw[6, :] =  [111,  74,   0]
    pallete_raw[7, :] =  [ 81,   0,  81]
    pallete_raw[8, :] =  [128,  64, 128]
    pallete_raw[9, :] =  [244,  35, 232]
    pallete_raw[10, :] =  [250, 170, 160]
    pallete_raw[11, :] = [230, 150, 140]
    pallete_raw[12, :] = [ 70,  70,  70]
    pallete_raw[13, :] = [102, 102, 156]
    pallete_raw[14, :] = [190, 153, 153]
    pallete_raw[15, :] = [180, 165, 180]
    pallete_raw[16, :] = [150, 100, 100]
    pallete_raw[17, :] = [150, 120,  90]
    pallete_raw[18, :] = [153, 153, 153]
    pallete_raw[19, :] = [153, 153, 153]
    pallete_raw[20, :] = [250, 170,  30]
    pallete_raw[21, :] = [220, 220,   0]
    pallete_raw[22, :] = [107, 142,  35]
    pallete_raw[23, :] = [152, 251, 152]
    pallete_raw[24, :] = [ 70, 130, 180]
    pallete_raw[25, :] = [220,  20,  60]
    pallete_raw[26, :] = [255,   0,   0]
    pallete_raw[27, :] = [  0,   0, 142]
    pallete_raw[28, :] = [  0,   0,  70]
    pallete_raw[29, :] = [  0,  60, 100]
    pallete_raw[30, :] = [  0,   0,  90]
    pallete_raw[31, :] = [  0,   0, 110]
    pallete_raw[32, :] = [  0,  80, 100]
    pallete_raw[33, :] = [  0,   0, 230]
    pallete_raw[34, :] = [119,  11,  32]

    train2regular = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

    for i in range(len(train2regular)):
        pallete[i, :] = pallete_raw[train2regular[i]+1, :]

    pallete = pallete.reshape(-1)

    return pallete

def main():

    # settings
    num_classes = 19
    snip_len = 30
    version = str(args.version)
    interv = args.interval
    num_ex = args.num_ex
    avg_acc = args.avg_acc

    # validate params
    if version not in ['18', '34', '50', '101']:
        raise ValueError("Invalid Accel version '%s' - must be one of Accel-{18,34,50,101}" % version)
    if interv < 1:
        raise ValueError("Invalid interval %d - must be >=1" % interv)
    if num_ex < 1:
        raise ValueError("Invalid num_ex %d - must be >=1" % num_ex)

    # get symbol
    pprint.pprint(config)
    config.symbol = 'accel_' + version
    model1 = '/../model/rfcn_dff_flownet_vid'
    model2 = '/../model/accel-' + version
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    key_sym = sym_instance.get_key_test_symbol(config)
    cur_sym = sym_instance.get_cur_test_symbol(config)

    path_demo_data = '/ebs/Accel/data/cityscapes/'
    path_demo_labels = '/ebs/Accel/data/cityscapes/'
    if path_demo_data == '' or path_demo_labels == '':
        raise ValueError("Must set path to demo data + labels")

    # load demo data
    image_names  = sorted(glob.glob(path_demo_data + 'leftImg8bit_sequence/val/frankfurt/*.png'))
    image_names += sorted(glob.glob(path_demo_data + 'leftImg8bit_sequence/val/lindau/*.png'))
    image_names += sorted(glob.glob(path_demo_data + 'leftImg8bit_sequence/val/munster/*.png'))
    image_names = image_names[: snip_len * num_ex]
    label_files  = sorted(glob.glob(path_demo_labels + 'gtFine/val/frankfurt/*trainIds.png'))
    label_files += sorted(glob.glob(path_demo_labels + 'gtFine/val/lindau/*trainIds.png'))
    label_files += sorted(glob.glob(path_demo_labels + 'gtFine/val/munster/*trainIds.png'))
    output_dir = cur_path + '/../demo/deeplab_dff/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    key_frame_interval = interv

    #
    lb_pos = 19
    image_names_trunc = []
    for i in range(num_ex):
        snip_pos = i * snip_len
        if avg_acc:
            offset = i % interv
        else:
            offset = interv - 1
        start_pos = lb_pos - offset
        image_names_trunc.extend(image_names[snip_pos + start_pos : snip_pos + start_pos + interv])
    image_names = image_names_trunc

    data = []
    key_im_tensor = None
    prev_im_tensor = None
    for idx, im_name in enumerate(image_names):
        assert os.path.exists(im_name), ('%s does not exist'.format(im_name))
        im = cv2.imread(im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        if idx % key_frame_interval == 0:
            key_im_tensor = im_tensor
        if prev_im_tensor is None:
            prev_im_tensor = im_tensor
        data.append({'data': im_tensor, 'im_info': im_info, 'data_key': prev_im_tensor, 'feat_key': np.zeros((1,config.network.DFF_FEAT_DIM,1,1))})
        prev_im_tensor = im_tensor

    # get predictor
    data_names = ['data', 'data_key', 'feat_key']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES]))),
                       ('data_key', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES]))),]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]

    arg_params, aux_params = load_param(cur_path + model1, 0, process=True)
    arg_params_dcn, aux_params_dcn = load_param(cur_path + model2, 0, process=True)
    arg_params.update(arg_params_dcn)
    aux_params.update(aux_params_dcn)
    key_predictor = Predictor(key_sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    cur_predictor = Predictor(cur_sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    nms = gpu_nms_wrapper(config.TEST.NMS, 0)

    # warm up
    for j in xrange(2):
        data_batch = mx.io.DataBatch(data=[data[j]], label=[], pad=0, index=0,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[j])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
        if j % key_frame_interval == 0:
            output_all, feat = im_segment(key_predictor, data_batch)
            output_all = [mx.ndarray.argmax(output['croped_score_output'], axis=1).asnumpy() for output in output_all]
        else:
            data_batch.data[0][-1] = feat
            data_batch.provide_data[0][-1] = ('feat_key', feat.shape)
            output_all, feat = im_segment(cur_predictor, data_batch)
            output_key = 'croped_score_output' if version == '101' else 'correction_output'
            output_all = [mx.ndarray.argmax(output[output_key], axis=1).asnumpy() for output in output_all]

    print "warmup done"
    # test
    time = 0
    count = 0
    hist = np.zeros((num_classes, num_classes))
    lb_idx = 0
    for idx, im_name in enumerate(image_names):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

        tic()
        if idx % key_frame_interval == 0:
            print '\n\nframe {} (key)'.format(idx)
            output_all, feat = im_segment(key_predictor, data_batch)
            output_all = [mx.ndarray.argmax(output['croped_score_output'], axis=1).asnumpy() for output in output_all]
        else:
            print '\nframe {} (intermediate)'.format(idx)
            data_batch.data[0][-1] = feat
            data_batch.provide_data[0][-1] = ('feat_key', feat.shape)
            output_all, feat = im_segment(cur_predictor, data_batch)
            output_key = 'croped_score_output' if version == '101' else 'correction_output'
            output_all = [mx.ndarray.argmax(output[output_key], axis=1).asnumpy() for output in output_all]

        elapsed = toc()
        time += elapsed
        count += 1
        print 'testing {} {:.4f}s [{:.4f}s]'.format(im_name, elapsed, time/count)

        pred = np.uint8(np.squeeze(output_all))
        segmentation_result = Image.fromarray(pred)
        pallete = getpallete(256)
        segmentation_result.putpalette(pallete)
        _, im_filename = os.path.split(im_name)
        segmentation_result.save(output_dir + '/seg_' + im_filename)

        # compute accuracy
        label = None

        _, lb_filename = os.path.split(label_files[lb_idx])
        im_comps = im_filename.split('_')
        lb_comps = lb_filename.split('_')
        # check if annotation available for frame
        if im_comps[1] == lb_comps[1] and im_comps[2] == lb_comps[2]:
            print 'label {}'.format(lb_filename)
            label = np.asarray(Image.open(label_files[lb_idx]))
            if lb_idx < len(label_files) - 1:
                lb_idx += 1

        if label is not None:
            curr_hist = fast_hist(pred.flatten(), label.flatten(), num_classes)
            hist += curr_hist
            print 'mIoU {mIoU:.3f}'.format(
                mIoU=round(np.nanmean(per_class_iu(curr_hist)) * 100, 2))
            print '(cum) mIoU {mIoU:.3f}'.format(
                mIoU=round(np.nanmean(per_class_iu(hist)) * 100, 2))

    ious = per_class_iu(hist) * 100
    print ' '.join('{:.03f}'.format(i) for i in ious)
    print '===> final mIoU {mIoU:.3f}'.format(mIoU=round(np.nanmean(ious), 2))

    print 'done'

if __name__ == '__main__':
    main()
