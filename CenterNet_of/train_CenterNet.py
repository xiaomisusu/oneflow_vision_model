from __future__ import print_function, absolute_import
import os
import sys
import time
import json
import datetime
import argparse
import os.path as osp
import oneflow as flow
import oneflow.nn as nn
import shutil
import oneflow.typing as tp
from typing import Tuple
import oneflow.math as math
import numpy as np

import tools.transforms.spatial_transforms as ST
import tools.transforms.temporal_transforms as TT
import tools.data_manager as data_manager

from tools.utils import AverageMeter, Logger

from tools.samplers import RandomIdentitySampler
import dla_model
from tools.opts import opts
# from util import Snapshot
# import cv2
np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='Train FairMOT')
# Datasets
parser.add_argument('--root', type=str, default='/data/MOT17/')
parser.add_argument('-d', '--dataset', type=str, default='MIX')
parser.add_argument('--height', type=int, default=608)
parser.add_argument('--width', type=int, default=1088)
parser.add_argument('--num_classes', type=int, default=80)
parser.add_argument('--data_cfg', default="/home/amax/gys/FairMOT_of/data/cfg/data.json", type=str)
# Augment
parser.add_argument("--model_load_dir", type=str, default='/home/amax/gys/FairMOT_of/pretrain/',
                    required=False,
                    help="model load directory")
parser.add_argument('--seq_len', type=int, default=4,
                    help="number of images to sample in a tracklet")
parser.add_argument('--sample_stride', type=int, default=8,
                    help="stride of images to sample in a tracklet")
# Optimization options
parser.add_argument('--max_epoch', default=30, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--train_batch', default=32, type=int)
parser.add_argument('--test_batch', default=1, type=int)
parser.add_argument('--lr', default=1.25e-4, type=float)
parser.add_argument('--stepsize', default=[20, 27], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--weight_decay', default=0.1, type=float)
parser.add_argument('--margin', type=float, default=0.3,
                    help="margin for triplet loss")
parser.add_argument('--distance', type=str, default='cosine',
                    help="euclidean or cosine")
parser.add_argument('--num_instances', type=int, default=4,
                    help="number of instances per identity")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='dla34',
                    help="ap3dres50, ap3dnlres50")
parser.add_argument('--K', type=int, default='128',
                    help="max number of output objects.")
parser.add_argument('--down_ratio', type=int, default='4',
                    help="")
parser.add_argument('--mse_loss', action='store_true',
                    help="use mse loss or focal loss to train")
parser.add_argument('--model_save_dir', default="/home/amax/gys/FairMOT_of/results/", type=str)

# Miscs
parser.add_argument('--eval_step', type=int, default=10)
parser.add_argument('--start_eval', type=int, default=0,
                    help="start to evaluate after specific epoch")
parser.add_argument('--save_dir', type=str, default='log-mars-ap3d')

parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
opts = opts().parse()
# config
func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
flow.config.gpu_device_num(1)
# def watch_image(y: tp.Numpy):
#     sub_img = y[0, 0, :, :]
#     sub_img = 1.0 / (1 + np.exp(-1 * sub_img))
#     sub_img = np.round(sub_img * 255)
#     cv2.imwrite('sub_image.jpg', sub_img)
#     print("out", y)
#
# def watch_image_sum(y: tp.Numpy):
#     sub_img = y[0, 0, :, :]
#     sub_img = 1.0 / (1 + np.exp(-1 * sub_img))
#     sub_img = np.round(sub_img * 255)
#     cv2.imwrite('sub_image.jpg', sub_img)
#     print("out", np.sum(y))
#
# def watch_dim1(y: tp.Numpy):
#     print("out", y)
#     print(y.shape)
#
# def watch_sum1(y: tp.Numpy):
#     print("out1", np.sum(y))
#
# def watch_sum2(y: tp.Numpy):
#     print("out2", np.sum(y))
#
# def watch_sum3(y: tp.Numpy):
#     print("out3", np.sum(y))
#
# def watch_sum4(y: tp.Numpy):
#     print("out4", np.sum(y))
#
# def watch_sum5(y: tp.Numpy):
#     print("out5", np.sum(y))
#
# def watch_sum6(y: tp.Numpy):
#     print("out6", np.sum(y))


def FocalLoss(pred, gt):

    pred = flow.clip(pred, min_value=1e-5, max_value=1-1e-5)

    one_temp = flow.ones(gt.shape, dtype=flow.float)

    pos_inds = flow.math.equal(gt, one_temp)
    pos_inds = flow.cast(pos_inds, dtype=flow.float)

    neg_inds = flow.math.less(gt, one_temp)
    neg_inds = flow.cast(neg_inds, dtype=flow.float)

    neg_weights = flow.math.pow(1 - gt, 4)

    pos_loss = flow.math.log(pred) * flow.math.pow(1 - pred, 2) * pos_inds
    neg_loss = flow.math.log(1 - pred) * flow.math.pow(pred, 2) * neg_weights * neg_inds

    num_pos = flow.math.reduce_sum(pos_inds, [0, 1, 2, 3])
    pos_loss = flow.math.reduce_sum(pos_loss, [0, 1, 2, 3])
    neg_loss = flow.math.reduce_sum(neg_loss, [0, 1, 2, 3])

    x = flow.clip(num_pos, 0, 1)
    x = flow.cast(x, dtype=flow.float)

    if_x_is_0 = (1-x) * (0 - neg_loss)
    if_x_is_1 = 0 - x * flow.math.divide(pos_loss+neg_loss, (num_pos+0.00001))

    return flow.math.add(if_x_is_0, if_x_is_1)

def RegL1Loss(output, mask, ind, target):
    # flow.watch(mask, watch_sum2)
    pred = _tranpose_and_gather_feat(output, ind)
    # flow.watch(pred, watch_sum5)
    mask = flow.expand_dims(mask, axis=2)

    mask_temp = mask
    for i in range(pred.shape[2] - 1):
        mask = flow.concat([mask, mask_temp], axis=2)

    #
    # flow.watch(pred*mask, watch_sum1)
    # flow.watch(target*mask, watch_sum6)
    loss = flow.nn.L1Loss(pred * mask, target * mask, reduction="sum")
    # flow.watch(loss, watch_sum3)
    for i in range(len(mask.shape)):
        mask = flow.math.reduce_sum(mask)

    loss = loss / (mask + 1e-4)
    return loss

def _tranpose_and_gather_feat(feat, ind):
    feat = flow.transpose(feat, [0, 2, 3, 1])
    feat = flow.reshape(feat, [feat.shape[0], -1, feat.shape[3]])
    feat = _gather_feat(feat, ind)
    return feat

def _gather_feat(feat, ind, mask=None):
    dim = feat.shape[2]
    ind = flow.expand_dims(ind, axis=2)

    ind_temp = ind
    for i in range(dim - 1):
        ind = flow.concat([ind, ind_temp], axis=2)

    ind = flow.cast(ind, dtype=flow.int32)
    feat = flow.dim_gather(feat, 1, ind)
    return feat

# inputs_size
input_image = tp.Numpy.Placeholder((opts.batch_size, 3, opts.input_h, opts.input_w))
input_heatmap = tp.Numpy.Placeholder((opts.batch_size, opts.num_classes, opts.input_h//opts.down_ratio, opts.input_w//opts.down_ratio))
input_reg_mask = tp.Numpy.Placeholder((opts.batch_size, opts.K))
input_ind = tp.Numpy.Placeholder((opts.batch_size, opts.K))
input_wh = tp.Numpy.Placeholder((opts.batch_size, opts.K, 2))
input_reg = tp.Numpy.Placeholder((opts.batch_size, opts.K, 2))
#ret = {'input': img, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg}

spatial_transform_train = ST.Compose([
        ST.Scale((opts.input_h, opts.input_w), interpolation=3),
        ST.RandomHorizontalFlip(),
        ST.ToNumpy(),
        ST.Normalize([0.40789654, 0.44719302, 0.47026115], [0.28863828, 0.27408164, 0.27809835])
    ])
dataset = data_manager.COCO(opts, "train", spatial_transform_train)

flow.config.enable_debug_mode(True)

def _norm(input):
    norm = flow.math.reduce_euclidean_norm(input, axis=1, keepdims=True)
    clamped = flow.clamp(norm, min_value=1e-12)
    denom = clamped
    for i in range(input.shape[1] - 1):
        denom = flow.concat([denom, clamped], axis=1)
    return flow.math.divide(input, denom)


@flow.global_function(type="train", function_config=func_config)
def train_job(image: input_image, hms: input_heatmap, reg_masks: input_reg_mask, inds: input_ind, whs: input_wh,
              regs: input_reg
              ) -> Tuple[tp.Numpy, tp.Numpy, tp.Numpy, tp.Numpy]: #-> tp.ListNumpy:
    # ret = {'input': img, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg}

    s_det = flow.ones([1], dtype=flow.float) * -1.85
    s_id = flow.ones([1], dtype=flow.float) * -1.05

    output = dla_model.CenterNet(image, args)

    hm_loss = FocalLoss(flow.math.sigmoid(output["hm"]), hms)
    wh_loss = RegL1Loss(output["wh"], reg_masks, inds, whs)
    off_loss = RegL1Loss(output["reg"], reg_masks, inds, regs)

    loss = 0.1*wh_loss + off_loss
    loss = 0.5 * loss

    lr_scheduler = flow.optimizer.PiecewiseScalingScheduler(
        base_lr=opts.lr, boundaries=opts.lr_step,
        scale=[0.1, 0.01]
    )
    flow.optimizer.Adam(lr_scheduler, do_bias_correction=False).minimize(loss)
    print("==> Loss computing finish.")
    return loss, hm_loss, wh_loss, off_loss


def train(epoch, dataset):
    batch_hm_loss = AverageMeter()
    batch_wh_loss = AverageMeter()
    batch_off_loss = AverageMeter()
    batch_id_loss = AverageMeter()
    batch_loss = AverageMeter()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    batch_size = opts.batch_size

    batch = 0
    indicies = [x for x in RandomIdentitySampler(dataset.num_samples)]
    for i in range(len(indicies) // batch_size):
        batch += 1
        try:
            train_batch = dataset.__getbatch__(indicies[i * batch_size:(i + 1) * batch_size])
        except:
            train_batch = dataset.__getbatch__(indicies[-batch_size:])
        data_time.update(time.time() - end)
        # ret = {'input': img, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg}
        loss, hm_loss, wh_loss, off_loss = train_job(train_batch[0], train_batch[1], train_batch[2],
                                                         train_batch[3], train_batch[4], train_batch[5])

        batch_hm_loss.update(hm_loss.item(), batch_size)
        batch_wh_loss.update(wh_loss.item(), batch_size)
        batch_off_loss.update(off_loss.item(), batch_size)
        batch_loss.update(loss.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()
        print('Epoch{0} '
              'batch{0} '
              'Time:{batch_time.sum:.1f}s '
              'Data:{data_time.sum:.1f}s '
              'Loss:{loss.avg:.4f} '
              'hm:{hm.avg:.4f} '
              'wh:{wh.avg:.4f} '
              'off:{off.avg:.4f} '
            .format(
            epoch + 1, batch, batch_time=batch_time, data_time=data_time, loss=batch_loss, hm=batch_hm_loss,
            wh=batch_wh_loss, off=batch_off_loss))

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
        .format(
        epoch + 1, batch_time=batch_time,
        data_time=data_time))


def main():
    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    checkpoint = flow.train.CheckPoint()
    if opts.load_model:
        assert os.path.isdir(args.model_load_dir)
        print("Restoring model from {}.".format(args.model_load_dir))
        checkpoint.load(args.model_load_dir)
    else:
        print("Init model")
        checkpoint.init()

    start_epoch = 0
    start_time = time.time()
    train_time = 0
    pre_epoch = -1

    print("==> Start training")
    for epoch in range(start_epoch, opts.num_epochs):
        start_train_time = time.time()
        train(epoch, dataset)
        train_time += round(time.time() - start_train_time)

        fpath = osp.join(opts.save_dir, 'checkpoint_ep' + str(epoch + 1))
        if os.path.exists(fpath):
            shutil.rmtree(fpath)
        if pre_epoch != -1:
            shutil.rmtree(osp.join(opts.save_dir, 'checkpoint_ep' + str(pre_epoch)))
        pre_epoch = epoch + 1
        checkpoint.save(fpath)

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


if __name__ == '__main__':
    main()

