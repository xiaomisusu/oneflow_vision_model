from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2

from tools.opts import opts
import oneflow as flow
import oneflow.typing as tp
import dla_model
import numpy as np
import tools.data_manager as data_manager
import PIL as Image

class MaxPool2D:
  def __init__(self, kernel_size=(2, 2), stride=2):
    self.stride = stride

    self.kernel_size = kernel_size
    self.w_height = kernel_size[0]
    self.w_width = kernel_size[1]

    self.x = None

    self.in_height = None
    self.in_width = None

    self.out_height = None
    self.out_width = None

    self.arg_max = None

  def __call__(self, x):
    self.x = x
    self.in_height = np.shape(x)[0]
    self.in_width = np.shape(x)[1]

    self.out_height = int((self.in_height - self.w_height) / self.stride) + 1
    self.out_width = int((self.in_width - self.w_width) / self.stride) + 1

    out = np.zeros((self.out_height, self.out_width))
    self.arg_max = np.zeros_like(out, dtype=np.int32)
    for i in range(self.out_height):
      for j in range(self.out_width):
        start_i = i * self.stride
        start_j = j * self.stride
        end_i = start_i + self.w_height
        end_j = start_j + self.w_width
        out[i, j] = np.max(x[start_i: end_i, start_j: end_j])
        self.arg_max[i, j] = np.argmax(x[start_i: end_i, start_j: end_j])

    self.arg_max = self.arg_max
    return out

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    # hmax = flow.nn.max_pool2d(input=heat, ksize=(kernel, kernel),
    #                           strides=1, padding=(0, 0, pad, pad))
    N, C, H, W = heat.shape
    max_pool_numpy = MaxPool2D((kernel, kernel), stride=1)
    input = heat
    input = input.reshape((N * C * H, W))
    input = np.pad(input, (pad, pad), 'constant')
    # input = input.reshape((N, C, H+pad*2, W+pad*2))
    hmax = max_pool_numpy(input)
    hmax = hmax.reshape((N, C, H, W))
    keep = (hmax == heat)
    keep = keep.astype(np.float)
    return heat * keep

def get_sorted_top_k(array, top_k=1, axis=-1, reverse=False):
  """
  多维数组排序
  Args:
      array: 多维数组
      top_k: 取数
      axis: 轴维度
      reverse: 是否倒序

  Returns:
      top_sorted_scores: 值
      top_sorted_indexes: 位置
  """
  # top_k = np.min(top_k, array.shape[axis])
  if reverse:
    # argpartition分区排序，在给定轴上找到最小的值对应的idx，partition同理找对应的值
    # kth表示在前的较小值的个数，带来的问题是排序后的结果两个分区间是仍然是无序的
    # kth绝对值越小，分区排序效果越明显
    axis_length = array.shape[axis]
    partition_index = np.take(np.argpartition(array, kth=-top_k, axis=axis),
                              range(axis_length - top_k, axis_length), axis)
  else:
    partition_index = np.take(np.argpartition(array, kth=top_k, axis=axis), range(0, top_k), axis)
  top_scores = np.take_along_axis(array, partition_index, axis)
  # 分区后重新排序
  sorted_index = np.argsort(top_scores, axis=axis)
  if reverse:
    sorted_index = np.flip(sorted_index, axis=axis)
  top_sorted_scores = np.take_along_axis(top_scores, sorted_index, axis)
  top_sorted_indexes = np.take_along_axis(partition_index, sorted_index, axis)
  return top_sorted_scores, top_sorted_indexes

def _topk(scores, K=40):
    batch, cat, height, width = scores.shape

    topk_scores, topk_inds = get_sorted_top_k(array=scores.reshape(batch, cat, -1), top_k=K, reverse=True)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).astype(np.int).astype(np.float)
    topk_xs = (topk_inds % width).astype(np.int).astype(np.float)

    topk_score, topk_ind = get_sorted_top_k(topk_scores.reshape(batch, -1), K, reverse=True)
    topk_clses = (topk_ind / K).astype(np.int)
    topk_inds = _gather_feat_np(topk_inds.reshape(batch, -1, 1), topk_ind).reshape(batch, K)
    topk_ys = _gather_feat_np(topk_ys.reshape(batch, -1, 1), topk_ind).reshape(batch, K)
    topk_xs = _gather_feat_np(topk_xs.reshape(batch, -1, 1), topk_ind).reshape(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def _sigmoid(x):
  y = flow.clamp(flow.math.sigmoid(x), min_value=1e-4, max_value=1-1e-4)
  return y

def _sigmoid12(x):
  y = flow.clamp(flow.math.sigmoid(x), 1e-12)
  return y

def _gather_feat(feat, ind):
  dim = feat.shape[2]
  # ind = flow.expand_dims(ind, axis=2)
  ind = flow.broadcast_like(ind,
                            like=flow.constant(value=0,
                                               dtype=ind.dtype,
                                               shape=(ind.shape[0], ind.shape[1], dim)),
                            broadcast_axes=(2,))
  # flow.watch(ind, watch_handler_ind)
  # flow.watch(feat, watch_handler_feat)

  feat = flow.dim_gather(feat, 1, ind)

  # flow.watch(feat, watch_handler_gather_feat)
  return feat

def gather_np(x, dim, index):
  """
  Gathers values along an axis specified by ``dim``.

  For a 3-D tensor the output is specified by:
      out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
      out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
      out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

  Parameters
  ----------
  dim:
      The axis along which to index
  index:
      A tensor of indices of elements to gather

  Returns
  -------
  Output Tensor
  """
  idx_xsection_shape = index.shape[:dim] + \
                        index.shape[dim + 1:]
  self_xsection_shape = x.shape[:dim] + x.shape[dim + 1:]
  if idx_xsection_shape != self_xsection_shape:
    raise ValueError("Except for dimension " + str(dim) +
                      ", all dimensions of index and self should be the same size")
  if index.dtype != np.dtype('int_'):
    raise TypeError("The values of index must be integers")
  if index.ndim == 3:
    first_indices = np.arange(index.shape[0])[:, None, None]
    second_indices = np.arange(index.shape[1])[None, :, None]
    third_indices = np.arange(index.shape[2])[None, None, :]
    if dim == 0:
      result = x[index, second_indices, third_indices]
    elif dim == 1:
      result = x[first_indices, index, third_indices]
    elif dim == 2:
      result = x[first_indices, second_indices, index]
  elif index.ndim == 4:
    first_indices = np.arange(index.shape[0])[:, None, None, None]
    second_indices = np.arange(index.shape[1])[None, :, None, None]
    third_indices = np.arange(index.shape[2])[None, None, :, None]
    fourth_indices = np.arange(index.shape[3])[None, None, None, :]
    if dim == 0:
      result = x[index, second_indices, third_indices, fourth_indices]
    elif dim == 1:
      result = x[first_indices, index, third_indices, fourth_indices]
    elif dim == 2:
      result = x[first_indices, second_indices, index, fourth_indices]
    elif dim == 3:
      result = x[first_indices, second_indices, third_indices, index]
  return result

def _gather_feat_np(feat, ind):
  dim = feat.shape[2]
  ind = np.expand_dims(ind, axis=2).repeat(dim, axis=2)
  feat = gather_np(feat, 1, ind)
  return feat

def _transpose_and_gather_feat(feat, ind):
  feat = flow.transpose(feat, [0, 2, 3, 1])
  feat = flow.reshape(feat, (feat.shape[0], -1, feat.shape[3]))
  feat = _gather_feat(feat, ind)
  return feat

def _transpose_and_gather_feat_np(feat, ind):
  feat = feat.transpose((0, 2, 3, 1))
  feat = np.reshape(feat, (feat.shape[0], -1, feat.shape[3]))
  feat = _gather_feat_np(feat, ind)
  return feat

def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.shape

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _transpose_and_gather_feat_np(reg, inds)
        reg = reg.reshape(batch, K, 2)
        xs = xs.reshape(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.reshape(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.reshape(batch, K, 1) + 0.5
        ys = ys.reshape(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat_np(wh, inds)
    if cat_spec_wh:
        wh = wh.reshape(batch, K, cat, 2)
        clses_ind = clses.reshape(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).reshape(batch, K, 2)
    else:
        wh = wh.reshape(batch, K, 2)
    clses = clses.reshape(batch, K, 1).astype(np.float)
    scores = scores.reshape(batch, K, 1)
    bboxes = np.concatenate([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], axis=2)
    detections = np.concatenate([bboxes, scores, clses], axis=2)

    return detections

@flow.global_function(flow.function_config())
def InferenceNet(images=flow.FixedTensorDef((1,3,512,512), dtype=flow.float)):
    output = dla_model.CenterNet(images, opt, training=False)
    hm = output['hm']
    hm = flow.math.sigmoid(hm)
    wh = output['wh']
    reg = output['reg']

    return hm, wh, reg

def image_preprocess(im):
    # im = im.resize((512, 512,3))
    base = np.ones([512, 512])
    mean = [0.40789654 * base, 0.44719302 * base, 0.47026115 * base]
    std = [0.28863828, 0.27408164, 0.27809835]

    im = cv2.resize(im, (512,512))
    im = np.transpose(im, (2, 0, 1))
    im = np.array(im).astype('float32')
    im = (im / 255.)
    im[0] = (im[0] - mean[0]) / std[0]
    im[1] = (im[1] - mean[1]) / std[1]
    im[2] = (im[2] - mean[2]) / std[2]

    im[[0,1,2], :, :] = im[[2,1,0],:,:]

    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, 'float32')

def convert_ret2cocoresult(ret, org_shape):
    result = []
    classes = ret[0, :, -1]
    top_preds = {}
    for i in range(opt.num_classes):
        inds = (classes == i)
        top_preds[i+1] = np.concatenate(
            [ret[0, inds, :4], ret[0, inds, 4:5]], axis=1)
    return top_preds

def test(opt):
  check_point = flow.train.CheckPoint()
  check_point.load(opt.load_model)

  opt = opts().update_dataset_info_and_set_heads(opt, data_manager.COCO)
  print(opt)


  split = 'val' if not opt.trainval else 'test'
  dataset = data_manager.COCO(opt, "test")

  results = {}
  num_iters = dataset.num_samples

  for ind in range(num_iters):
    img_id = dataset.images[ind]
    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])

    org_img = cv2.imread(img_path)
    img = image_preprocess(org_img)
    output = InferenceNet(img).get()
    ret = ctdet_decode(output[0], output[1], output[2], K=opt.K)

    ret = convert_ret2cocoresult(ret, org_img.shape)
    results[img_id] = ret
    print("Processing: {}/{}.".format(ind, num_iters))

  dataset.run_eval(results, opt.save_dir)

if __name__ == '__main__':
  opt = opts().parse()

  test(opt)
