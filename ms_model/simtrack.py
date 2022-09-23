# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import numpy as np
import copy
import logging
from collections import defaultdict
import mindspore
import torch
from simtrack.det3d.core import box_torch_ops
# from ms_sim import box_torch_ops
from mindspore import nn, ops, Tensor, numpy
from mindspore import numpy as mnp
from mindspore.common import dtype as mstype
from ms_sim.ms_model.bbox_heads.loss import FastFocalLoss, RegLoss

def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor"""

    actual_num = ops.ExpandDims()(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = mnp.arange(0, max_num, dtype=mstype.int32).view(*max_num_shape)
    paddings_indicator = actual_num > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class PFNLayer(nn.Cell):
    """
   Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.

    Args:
        in_channels (int): Input channel.
        out_channels (int): Output channel.
        use_norm:
        last_layer:<bool>. If last_layer, there is no concatenation of features.

    Returns:
        Tensor, output tensor.
    """

    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        super(PFNLayer, self).__init__()

        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.linear = nn.Dense(in_channels, self.units, has_bias=False)
        if norm_cfg is None:
            self.norm = nn.BatchNorm2d(self.units)#, eps=1e-3, momentum=0.99

        self.transpose = ops.Transpose()
        self.tile = ops.Tile()
        self.concat = ops.Concat(axis=2)
        self.expand_dims = ops.ExpandDims()
        self.argmax_w_value = ops.ArgMaxWithValue(axis=1, keep_dims=True)

    def construct(self, inputs):
        """forward graph"""
        x = self.linear(inputs)
        x = self.expand_dims(x, 3)
        x = x.transpose((0, 2, 1, 3))
        x = self.norm(x)
        x = x.transpose((0, 2, 1, 3)).squeeze(axis=3)
        x = ops.ReLU()(x)
        x_max = self.argmax_w_value(x)[1]
        if self.last_vfe:
            return x_max
        x_repeat = self.tile(x_max, (1, inputs.shape[1], 1))
        x_concatenated = self.concat([x, x_repeat])
        return x_concatenated


class PillarFeatureNet(nn.Cell):
    """Pillar feature net
    the network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
    similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
    :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
    :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
    :param with_distance: <bool>. Whether to include Euclidean distance to points.
    :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
    :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
    """

    def __init__(
            self,
            num_input_features,
            norm_cfg,
            num_filters,
            with_distance,
            voxel_size,
            pc_range
    ):
        super().__init__()
        self.name = "PillarFeatureNet"
        assert len(num_filters) > 0
        num_input_features += 5

        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []

        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True

            pfn_layers.append(
                PFNLayer(in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer)
            )
        self.pfn_layers = nn.SequentialCell(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]
        self.expand_dims = ops.ExpandDims()

    def construct(self, features, num_points, coors):
        """forward graph"""

        points_mean = (features[:, :, :3].sum(axis=1, keepdims=True) /
                       ops.Maximum()(num_points, 1).view(-1, 1, 1))
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = ops.ZerosLike()(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
                self.expand_dims(coors[:, 3].astype(mstype.float32), 1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (
                self.expand_dims(coors[:, 2].astype(mstype.float32), 1) * self.vy + self.y_offset)

        # Combine feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = mnp.norm(features[:, :, :3], 2, 2, keepdims=True)
            features_ls.append(points_dist)
        features = ops.Concat(axis=-1)(features_ls)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zero.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = self.expand_dims(mask, -1).astype(features.dtype)
        features *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)
        return features.squeeze()


class PointPillarsScatter(nn.Cell):
    """PointPillars scatter
    Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
    second.pytorch.voxelnet.SparseMiddleExtractor.
    :param output_shape: ([int]: 4). Required output shape of features.
    :param num_input_features: <int>. Number of input features."""

    def __init__(self, num_input_features, norm_cfg, name="PointPillarsScatter", **kwargs):
        super().__init__()
        self.name = "PointPillarsScatter"
        self.nchannels = num_input_features

    def construct(self, voxel_features, coords, batch_size, input_shape):
        """forward graph"""
        self.ny = input_shape[0]
        self.nx = input_shape[1]
        # Batch_canvas will be the final output.

        # z coordinate is not used, z -> batch
        batch_canvas = []
        for batch_itt in range(batch_size):  # [bs, v, p, 64]
            canvas = ops.zeros((int(self.nchannels), int(self.nx * self.ny)), mindspore.float32)
            batch_mask = (coords[:, 0] == batch_itt)*1
            batch_mask = batch_mask.nonzero().squeeze(axis=1)
            this_coords = coords.gather(batch_mask, axis=0)
            indices = (this_coords[:, 2] * self.nx + this_coords[:, 3]).astype('int')
            voxels = voxel_features.gather(batch_mask, axis=0)
            voxels = voxels.T

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        stack = ops.Stack()
        batch_canvas = stack(batch_canvas)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, int(self.ny), int(self.nx))
        return batch_canvas


class RPN(nn.Cell):
    def __init__(
        self,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        us_layer_strides,
        us_num_filters,
        num_input_features,
        norm_cfg=None,
        name="rpn",
        logger=None,
    ):
        super(RPN, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters
        self._num_input_features = num_input_features

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        must_equal_list = []
        for i in range(len(self._upsample_strides)):
            must_equal_list.append(
                self._upsample_strides[i]
                / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1])
            )

        for val in must_equal_list:
            assert val == must_equal_list[0]

        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                stride=self._layer_strides[i],
            )
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = (self._upsample_strides[i - self._upsample_start_idx])
                if stride >= 1:
                    deblock = nn.SequentialCell(
                        nn.Conv2dTranspose(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            pad_mode='same',
                            stride=stride,
                            has_bias=False),
                        nn.BatchNorm2d(
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            eps=1e-3, momentum=0.99
                        ),
                        nn.ReLU(),
                    )
                else:
                    stride = int(np.round(1 / stride))
                    deblock = nn.SequentialCell(
                        nn.Conv2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            kernel_size=stride,
                            pad_mode='valid',
                            stride=stride,
                            has_bias=False),
                        nn.BatchNorm2d(
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            eps=1e-3, momentum=0.99
                        ),
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self.blocks = nn.CellList(blocks)
        self.deblocks = nn.CellList(deblocks)

        logger.info("Finish RPN Initialization")

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        block = nn.SequentialCell(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(int(inplanes), planes, 3, stride=stride, pad_mode='valid', has_bias=False),
            nn.BatchNorm2d(planes, eps=1e-3, momentum=0.99),
            nn.ReLU(),
        )

        for j in range(num_blocks):
            block.append(nn.Conv2d(planes, planes, 3, padding=1, pad_mode='pad', has_bias=False))
            block.append(nn.BatchNorm2d(planes, eps=1e-3, momentum=0.99))
            block.append(nn.ReLU())

        return block, planes

    def construct(self, x):
        ups = []
        op = ops.Concat(1)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = op((ups[0], ups[1], ups[2]))
        return x


class SepHead(nn.Cell):
    def __init__(
            self,
            in_channels,
            heads,
            head_conv=64,
            final_kernel=1,
            bn=False,
            init_bias=-2.19,
            **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)

        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]
            fc = nn.SequentialCell()
            if 'hm' in head:
                for i in range(num_conv - 1):
                    fc.append(nn.Conv2d(in_channels, head_conv,
                                 kernel_size=final_kernel, padding=final_kernel // 2,
                                 pad_mode='pad', has_bias=True))
                    if bn:
                        fc.append(nn.BatchNorm2d(head_conv, momentum=0.90))
                    fc.append(nn.ReLU())
                fc.append(nn.Conv2d(head_conv, classes,
                        kernel_size=final_kernel, padding=final_kernel // 2, pad_mode='pad', has_bias=True))
            else:
                for i in range(num_conv - 1):
                    fc.append(nn.Conv2d(in_channels, head_conv,
                                        kernel_size=final_kernel, padding=final_kernel // 2, weight_init='HeUniform',
                                        pad_mode='pad', has_bias=True))
                    if bn:
                        fc.append(nn.BatchNorm2d(head_conv, momentum=0.90))
                    fc.append(nn.ReLU())
                fc.append(nn.Conv2d(head_conv, classes,
                                    kernel_size=final_kernel, padding=final_kernel // 2, weight_init='HeUniform',
                                    pad_mode='pad', has_bias=True))

            self.__setattr__(head, fc)

    def construct(self, x):
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


class CenterHeadV2(nn.Cell):
    def __init__(
            self,
            in_channels,
            tasks,
            weight,
            code_weights,
            common_heads,
            logger=None,
            init_bias=-2.19,
            share_conv_channel=64,
            num_hm_conv=2,
    ):
        super(CenterHeadV2, self).__init__()

        num_classes = []
        for t in tasks:
            num_classes.append(len(t["class_names"]))
        self.class_names = [t["class_names"] for t in tasks]
        self.code_weights = code_weights
        self.weight = weight  # weight between hm loss and loc loss

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()

        if not logger:
            logger = logging.getLogger("CenterHead")
        self.logger = logger

        logger.info(
            f"num_classes: {num_classes}"
        )

        # a shared convolution
        self.shared_conv = nn.SequentialCell(
            nn.Conv2d(in_channels, share_conv_channel,
                      kernel_size=3, padding=1, pad_mode='pad', has_bias=True),
            nn.BatchNorm2d(share_conv_channel),#, momentum=0.9),
            nn.ReLU()
        )

        self.tasks = nn.CellList([])
        print("Use HM Bias: ", init_bias)

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(hm=(num_cls, num_hm_conv)))
            self.tasks.append(
                SepHead(share_conv_channel, heads, bn=True, init_bias=init_bias, final_kernel=3)
            )

        logger.info("Finish CenterHead Initialization")

    def construct(self, x):
        ret_dicts = []

        x = self.shared_conv(x)
        for task in self.tasks:
            ret_dicts.append(task(x))

        return ret_dicts

    def _sigmoid(self, x):
        min_value = Tensor(1e-4, mindspore.float32)
        max_value = Tensor(1-1e-4, mindspore.float32)
        sigmoid = nn.Sigmoid()
        y = ops.clip_by_value(sigmoid(x), min_value, max_value)
        return y

    def loss(self, example, preds_dicts, **kwargs):
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])

            hm_loss = self.crit(preds_dict['hm'], example['hm'][task_id], example['ind'][task_id], example['mask'][task_id], example['cat'][task_id])

            target_box = example['anno_box'][task_id]
            # reconstruct the anno_box from multiple reg heads
            cat = ops.Concat(axis=1)
            preds_dict['anno_box'] = cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                preds_dict['vel'], preds_dict['rot']))

            ret = {}

            # Regression loss for dimension, offset, height, rotation
            box_loss = self.crit_reg(preds_dict['anno_box'], example['mask'][task_id], example['ind'][task_id], target_box)

            loc_loss = (box_loss * ops.stop_gradient(self.code_weights.copy())).sum()

            loss = hm_loss + self.weight * loc_loss

            ret.update({'loss': loss, 'hm_loss': ops.stop_gradient(hm_loss), 'loc_loss': loc_loss,
                        'loc_loss_elem': ops.stop_gradient(box_loss), 'num_positive': example['mask'][task_id].sum()})

            rets.append(ret)

        """convert batch-key to key-batch
        """
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        return rets_merged


    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        """decode, nms, then return the detection result. Additionaly support double flip testing
        """
        # get loss info
        rets = []
        metas = []

        double_flip = test_cfg.get('double_flip', False)

        post_center_range = test_cfg['post_center_limit_range']
        if len(post_center_range) > 0:
            post_center_range = Tensor(post_center_range)

        for task_id, preds_dict in enumerate(preds_dicts):
            # convert N C H W to N H W C
            for key, val in preds_dict.items():
                preds_dict[key] = val.transpose((0, 2, 3, 1))

            # batch_size = preds_dict['hm'].shape[0]

            # if "metadata" not in example or len(example["metadata"]) == 0:
            #     meta_list = [None] * batch_size
            # else:
            #     meta_list = example["metadata"]
            #     if double_flip:
            #         meta_list = meta_list[: 4 *int(batch_size):4]

            meta_list = example['token']


            sigmoid = ops.Sigmoid()
            exp = ops.Exp()
            atan2 = ops.Atan2()
            batch_hm = sigmoid(preds_dict['hm'])
            batch_dim = exp(preds_dict['dim'])
            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            batch_rot = atan2(batch_rots, batch_rotc)

            batch, H, W, num_cls = batch_hm.shape

            reshape = ops.Reshape()
            batch_reg = reshape(batch_reg, (batch, H * W, 2))
            batch_hei = reshape(batch_hei, (batch, H * W, 1))
            batch_rot = reshape(batch_rot, (batch, H * W, 1))
            batch_dim = reshape(batch_dim, (batch, H * W, 3))
            batch_hm = reshape(batch_hm, (batch, H * W, num_cls))

            meshgrid = ops.Meshgrid(indexing="ij")
            cat = ops.Concat(2)

            ys, xs = meshgrid((numpy.arange(0, H), numpy.arange(0, W)))

            ys = numpy.tile(ys.view(1, H, W), (batch, 1, 1))
            xs = numpy.tile(xs.view(1, H, W), (batch, 1, 1))

            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

            xs = xs * test_cfg['out_size_factor'] * test_cfg['voxel_size'][0] + test_cfg['pc_range'][0]
            ys = ys * test_cfg['out_size_factor'] * test_cfg['voxel_size'][1] + test_cfg['pc_range'][1]

            batch_vel = preds_dict['vel']

            batch_vel = reshape(batch_vel, (batch, H * W, 2))
            batch_box_preds = cat([xs, ys, batch_hei, batch_dim, batch_vel, batch_rot])

            metas.append(meta_list)

            rets.append(self.post_processing(batch_box_preds, batch_hm, test_cfg, post_center_range))

            # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        concat = ops.Concat()
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    ret[k] = concat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = concat([ret[i][k] for ret in rets])

            ret['token'] = metas[0]
            ret_list.append(ret)

        return ret_list


    def post_processing(self, batch_box_preds, batch_hm, test_cfg, post_center_range):
        batch_size = len(batch_hm)

        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]

            labels, scores = ops.ArgMaxWithValue(axis=-1)(hm_preds)

            score_mask = scores > test_cfg['score_threshold']
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1).asnumpy() \
                            & (box_preds[..., :3] <= post_center_range[3:]).all(1).asnumpy()

            mask = distance_mask & score_mask.asnumpy()

            mask = (Tensor(mask, mindspore.bool_)*1).nonzero().squeeze(axis=1)
            box_preds = box_preds.gather(mask, axis=0)
            scores = scores.gather(mask, axis=0)
            labels = labels.gather(mask, axis=0)


            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

            boxes_for_nms = torch.tensor(boxes_for_nms.asnumpy()).to('cuda')
            scores = torch.tensor(scores.asnumpy()).to('cuda')

            selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms, scores,
                                                      thresh=test_cfg['nms']['nms_iou_threshold'],
                                                      pre_maxsize=test_cfg['nms']['nms_pre_max_size'],
                                                      post_max_size=test_cfg['nms']['nms_post_max_size'])


            selected = mindspore.Tensor(selected.to('cpu').numpy(), mindspore.int64)
            boxes_for_nms = mindspore.Tensor(boxes_for_nms.to('cpu').numpy(), mindspore.float32)
            scores = mindspore.Tensor(scores.to('cpu').numpy(), mindspore.float32)

            selected_boxes = box_preds.gather(selected, axis=0)
            selected_scores = scores.gather(selected, axis=0)
            selected_labels = labels.gather(selected, axis=0)

            prediction_dict = {
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels
            }

            prediction_dicts.append(prediction_dict)

        return prediction_dicts


    def predict_tracking(self, example, preds_dicts, test_cfg, **kwargs):
        """decode, nms, then return the detection result.
        """
        rets = []
        metas = []
        post_center_range = test_cfg['post_center_limit_range']
        if len(post_center_range) > 0:
            post_center_range = Tensor(post_center_range, mindspore.float32)

        prev_track_id = kwargs.get('prev_track_id', None)

        if prev_track_id is not None:
            track_rets = []
            prev_hm = kwargs['prev_hm']

        for task_id, preds_dict in enumerate(preds_dicts):
            new_obj = [{}]
            # convert N C H W to N H W C
            for key, val in preds_dict.items():
                preds_dict[key] = val.transpose((0, 2, 3, 1))

            batch_size = preds_dict['hm'].shape[0]

            # if "metadata" not in example or len(example["metadata"]) == 0:
            #     meta_list = [None] * batch_size
            # else:
            meta_list = example["token"]

            ######################################################
            sigmoid = ops.Sigmoid()
            exp = ops.Exp()
            atan2 = ops.Atan2()
            batch_hm = sigmoid(preds_dict['hm'])
            batch_dim = exp(preds_dict['dim'])
            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            batch_rot = atan2(batch_rots, batch_rotc)

            batch, H, W, num_cls = batch_hm.shape

            reshape = ops.Reshape()
            batch_reg = reshape(batch_reg, (batch, H * W, 2))
            batch_hei = reshape(batch_hei, (batch, H * W, 1))
            batch_rot = reshape(batch_rot, (batch, H * W, 1))
            batch_dim = reshape(batch_dim, (batch, H * W, 3))
            batch_hm = reshape(batch_hm, (batch, H * W, num_cls))

            meshgrid = ops.Meshgrid(indexing="ij")
            expand_dims = ops.ExpandDims()
            cat = ops.Concat(2)
            ys, xs = meshgrid((numpy.arange(0, H), numpy.arange(0, W)))
            ys = expand_dims(ys, 0)
            xs = expand_dims(xs, 0)
            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

            xs = xs * test_cfg['out_size_factor'] * test_cfg['voxel_size'][0] + test_cfg['pc_range'][0]
            ys = ys * test_cfg['out_size_factor'] * test_cfg['voxel_size'][1] + test_cfg['pc_range'][1]

            batch_vel = preds_dict['vel']

            batch_vel = reshape(batch_vel, (batch, H * W, 2))
            batch_box_preds = cat([xs, ys, batch_hei, batch_dim, batch_vel, batch_rot])

            metas.append(meta_list)

            if prev_track_id is not None:
                tracking_batch_hm = (batch_hm + prev_hm[task_id]) / 2.0
                tracking = self.post_processing_tracking(batch_box_preds, tracking_batch_hm, prev_track_id[task_id],
                                                         test_cfg, post_center_range)

                for bit in range(len(tracking)):
                    cond = tracking[bit]['tracking_id'] == -1  # new obj
                    for tk in tracking[0].keys():
                        if tk != 'tracking_id':
                            new_obj[bit][tk] = tracking[bit][tk][cond]

                for bit in range(len(tracking)):
                    cond = tracking[bit]['tracking_id'] != -1
                    for tk in tracking[0].keys():
                        tracking[bit][tk] = tracking[bit][tk][cond]

                track_rets.append(tracking)

            else:
                new_obj = self.post_processing(batch_box_preds, batch_hm, test_cfg, post_center_range)

            rets.append(new_obj)

        # Merge branches results
        ret_list = []
        concat = ops.Concat()
        num_samples = len(rets[0])
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores", "selected_id"]:
                    ret[k] = concat([retss[i][k] for retss in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class

                    ret[k] = concat([retss[i][k] for retss in rets])

            ret['metadata'] = metas[0][i]
            ret_list.append(ret)

        if prev_track_id is not None:
            track_rets_list = []
            num_tracks = len(track_rets[0])
            for i in range(num_tracks):
                ret = {}
                for k in ['box3d_lidar', 'scores', 'label_preds', 'tracking_id']:
                    if k in ["box3d_lidar", "scores", 'tracking_id']:
                        ret[k] = concat([retss[i][k] for retss in track_rets])
                    elif k in ["label_preds"]:
                        flag = 0
                        for j, num_class in enumerate(self.num_classes):
                            track_rets[j][i][k] += flag
                            flag += num_class
                        ret[k] = concat([retss[i][k] for retss in track_rets])

                ret['metadata'] = metas[0][i]
                track_rets_list.append(ret)

            return ret_list, track_rets_list
        else:
            return ret_list


    def post_processing_tracking(self, batch_box_preds, batch_hm, prev_tracking_id, test_cfg, post_center_range):
        batch_size = len(batch_hm)

        prediction_dicts = []
        expand_dims = ops.ExpandDims()
        squeeze = ops.Squeeze(-1)
        for i in range(batch_size):
            box_preds = batch_box_preds[i].copy()
            hm_preds = batch_hm[i].copy()
            prev_id = prev_tracking_id[i]
            labels, scores = ops.ArgMaxWithValue(axis=-1)(hm_preds)
            prev_id = squeeze(ops.GatherD()(prev_id, 1, expand_dims(labels, -1)))

            score_mask = scores > test_cfg['score_threshold']
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1).asnumpy() \
                            & (box_preds[..., :3] <= post_center_range[3:]).all(1).asnumpy()

            mask = distance_mask & score_mask.asnumpy()
            mask = (Tensor(mask, mindspore.bool_) * 1).nonzero().squeeze(axis=1)

            box_preds = box_preds.gather(mask, axis=0)
            scores = scores.gather(mask, axis=0)
            labels = labels.gather(mask, axis=0)
            prev_id = prev_id.gather(mask, axis=0)

            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

            boxes_for_nms = torch.tensor(boxes_for_nms.asnumpy()).to('cuda')
            scores = torch.tensor(scores.asnumpy()).to('cuda')

            selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms, scores,
                                                      thresh=test_cfg['nms']['nms_iou_threshold'],
                                                      pre_maxsize=test_cfg['nms']['nms_pre_max_size'],
                                                      post_max_size=test_cfg['nms']['nms_post_max_size'])


            selected = mindspore.Tensor(selected.to('cpu').numpy(), mindspore.int64)
            boxes_for_nms = mindspore.Tensor(boxes_for_nms.to('cpu').numpy(), mindspore.float32)
            scores = mindspore.Tensor(scores.to('cpu').numpy(), mindspore.float32)


            selected_boxes = box_preds.gather(selected, axis=0)
            selected_scores = scores.gather(selected, axis=0)
            selected_labels = labels.gather(selected, axis=0)
            selected_id = prev_id.gather(selected, axis=0)

        prediction_dict = {
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels,
                "tracking_id": selected_id,
            }

        prediction_dicts.append(prediction_dict)

        return prediction_dicts


class SimtrackNet(nn.Cell):
    def __init__(
            self,
            PFN_num_input_features,
            num_filters,
            with_distance,
            voxel_size,
            pc_range,
            norm_cfg,
            PPS_num_input_features,
            layer_nums,
            ds_layer_strides,
            ds_num_filters,
            us_layer_strides,  # #[1, 2, 4], #,
            us_num_filters,
            RPN_num_input_features,
            in_channels,  # this is linked to 'neck' us_num_filters
            tasks,
            weight,
            code_weights,
            common_heads,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
    ):
        super().__init__()

        self.test = test_cfg

        self.reader = PillarFeatureNet(
            PFN_num_input_features,
            norm_cfg,
            num_filters,
            with_distance,
            voxel_size,
            pc_range
        )

        self.backbone = PointPillarsScatter(
            PPS_num_input_features, norm_cfg)

        self.neck = RPN(
            layer_nums,
            ds_layer_strides,
            ds_num_filters,
            us_layer_strides,  # #[1, 2, 4], #,
            us_num_filters,
            RPN_num_input_features,
            norm_cfg,
            logger=logging.getLogger("RPN")
        )

        self.bbox_head = CenterHeadV2(
            in_channels,  # this is linked to 'neck' us_num_filters
            tasks,
            weight,
            code_weights,
            common_heads
        )

    def extract_feat(self, data):
        input_features = self.reader(data["features"], data["num_voxels"], data["coors"])

        x = self.backbone(input_features, data["coors"], data["batch_size"], data["input_shape"])

        x = self.neck(x)
        return x

    def construct(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)

        preds = self.bbox_head(x)

        return_feature = kwargs.get('return_feature', False)
        if return_feature:
            return preds
        if return_loss:
            return self.bbox_head.loss(example, preds)
        else:
            return self.bbox_head.predict(example, preds, self.test)


if __name__ == '__main__':
    from mindspore import context
    import mindspore as ms
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')

    num_input_features = 64
    net = PointPillarsScatter(num_input_features, norm_cfg=None, ds_factor=1)
    voxel_features = ms.Tensor(np.ones([148139, 64]), ms.float32)
    voxel_features[0, :] = 0
    coords = ms.Tensor(np.ones([148139, 4]), ms.int32)
    coords[0, :] = 0
    coords[2, :] = 2
    coords[3, :] = 3
    coords[4, :] = 4
    coords[5, :] = 5
    coords[6, :] = 6
    coords[7, :] = 7
    input_shape = [512, 512, 1]
    batch_size = 8
    output = net(voxel_features, coords, batch_size, input_shape)  # Tensor(shape=[2, 64, 248, 296], dtype=Float32
    print(output.shape)