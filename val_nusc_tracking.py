import argparse
import copy
import os
import pickle

import mindspore
import numpy as np
import torch
from ms_sim.datasets import build_dataset
import simtrack.det3d.torchie.trainer.checkpoint as py_check
# from simtrack.det3d.datasets import build_dataloader
# from simtrack.det3d.datasets import build_dataloader, build_dataset
# from simtrack.det3d.models import build_detector
from simtrack.det3d.torchie import Config

# from simtrack.det3d.torchie.trainer import load_checkpoint
from ms_sim.torchie.trainer.trainer import example_to_device
from simtrack.det3d.torchie.trainer.utils import all_gather, synchronize
from simtrack.det3d.core.utils.center_utils import (draw_gaussian, gaussian_radius)
from simtrack.det3d.core.bbox.box_np_ops import center_to_corner_box2d
from simtrack.det3d.core.bbox.geometry import points_in_convex_polygon_jit

from nuscenes.nuscenes import NuScenes
from ms_sim.ms_model.simtrack import SimtrackNet
from mindspore import load_checkpoint, ops, Tensor
from mindspore import load_param_into_net
from mindspore import dataset as de
from mindspore import context

def parse_args():
    parser = argparse.ArgumentParser(description="Nuscenes Tracking")
    parser.add_argument("config", help="train config file path",
                        default='examples/point_pillars/configs/nusc_all_pp_centernet_tracking.py')
    parser.add_argument("work_dir", help="the dir to save logs and models", default='SAVE_DIR')
    parser.add_argument(
        "checkpoint", help="the dir to checkpoint which the model read from",
        default='/home/user/zad/simtrack/model_zoo/simtrack_pillar.pth')
    parser.add_argument("local_rank", type=int, default=0)
    parser.add_argument("eval_det", action='store_true')

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def tracking(args):
    # args = parse_args()
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    global voxel_size, downsample, voxel_range, num_classes, size_h, size_w
    voxel_size = np.array(cfg._voxel_size)[:2]
    downsample = cfg.assigner.out_size_factor
    voxel_range = np.array(cfg._pc_range)
    num_classes = sum([t['num_class'] for t in cfg.tasks])
    size_w, size_h = ((voxel_range[3:5] - voxel_range[:2]) / voxel_size / downsample).astype(np.int32)

    dataset = build_dataset(cfg.data.val)


    ms_model = SimtrackNet(PFN_num_input_features=5,
                           num_filters=[64, 64],
                           with_distance=False,
                           voxel_size=(0.2, 0.2, 8),
                           pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                           norm_cfg=None,
                           PPS_num_input_features=64,
                           layer_nums=[3, 5, 5],
                           ds_layer_strides=[2, 2, 2],
                           ds_num_filters=[64, 128, 256],
                           us_layer_strides=[0.5, 1, 2],  # #[1, 2, 4], #,
                           us_num_filters=[128, 128, 128],
                           RPN_num_input_features=64,
                           in_channels=sum([128, 128, 128]),  # this is linked to 'neck' us_num_filters
                           tasks=[
                               dict(num_class=1, class_names=["car"], stride=1),
                               dict(num_class=2, class_names=["truck", "construction_vehicle"], stride=1),
                               dict(num_class=2, class_names=["bus", "trailer"], stride=1),
                               dict(num_class=1, class_names=["barrier"], stride=1),
                               dict(num_class=2, class_names=["motorcycle", "bicycle"], stride=1),
                               dict(num_class=2, class_names=["pedestrian", "traffic_cone"], stride=1)],
                           weight=0.25,
                           code_weights=[4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
                           common_heads={'reg': (2, 2), 'height': (1, 2), 'dim': (3, 2), 'rot': (2, 2), 'vel': (2, 2)},
                           train_cfg=None,
                           test_cfg={
                               'nms': {'nms_pre_max_size': 1000, 'nms_post_max_size': 83, 'nms_iou_threshold': 0.2},
                               'score_threshold': 0.1, 'pc_range': [-51.2, -51.2], 'out_size_factor': 4,
                               'voxel_size': [0.2, 0.2],
                               'post_center_limit_range': [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0], 'max_per_img': 500},
                           pretrained=None)


    from simtrack.det3d.models import build_detector
    torch_model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    torch_checkpoint = torch.load('/home/zad/project/simtrack/model_zoo/simtrack_pillar.pth')
    state_dict = torch_checkpoint["state_dict"]
    _ = py_check.load_checkpoint(torch_model, '/home/zad/project/simtrack/model_zoo/simtrack_pillar.pth', map_location="cpu")

    eval_column_names = ['token', 'points', 'voxels', 'shape', 'num_points', 'num_voxels', 'coordinates', 'ref_from_car', 'car_from_global']
    ds = de.GeneratorDataset(dataset, column_names=eval_column_names, shuffle=False)
    detections = {}

    ms_checkpoint = load_checkpoint('/home/zad/project/ms_sim/simtrack_pillar.ckpt')
    ms_checkpoint.items()
    load_param_into_net(ms_model, ms_checkpoint)

    prev_detections = {}
    nusc = NuScenes(version='v1.0-mini', dataroot='/home/zad/project/v1.0-mini', verbose=True)
    grids = meshgrid(size_w, size_h)

    expand_dims = ops.ExpandDims()

    start_id = 0

    for _, data_batch in enumerate(ds.create_dict_iterator()):

        data_batch = example_to_device(data_batch, non_blocking=False)

        points = ops.zeros((data_batch['points'].shape[0], 6), mindspore.float32)
        points[:, 1:6] = data_batch['points']
        data_batch['points'] = points
        coor = ops.zeros((data_batch['coordinates'].shape[0], 4), mindspore.float32)
        coor[:, 1:4] = data_batch['coordinates']
        data_batch['coordinates'] = coor
        data_batch['shape'] = data_batch['shape'].view(1, 3)
        data_batch['ref_from_car'] = expand_dims(data_batch['ref_from_car'], 0)
        data_batch['car_from_global'] = expand_dims(data_batch['car_from_global'], 0)


        prev_token = nusc.get('sample', str(data_batch['token']))['prev']
        track_outputs = None
        transpose = ops.Transpose()
        if prev_token != '':  # non-first frame
            assert prev_token in prev_detections.keys()
            box3d = prev_detections[prev_token]['box3d_global']
            box3d = (data_batch['ref_from_car'][0].asnumpy() @ data_batch['car_from_global'][
                0].asnumpy()) @ box3d
            box3d = box3d.T
            prev_detections[prev_token]['box3d_lidar'] = np.concatenate((box3d[:, :3],
                                                                         prev_detections[prev_token]['box3d_lidar'][
                                                                         :, 3:]), axis=1)

            prev_hm_, prev_track_id_ = render_trackmap(prev_detections[prev_token], grids, cfg)
            prev_hm_ = prev_hm_.transpose((0, 2, 3, 1)).view(1, int(size_h * size_w), int(num_classes))
            prev_track_id_ = prev_track_id_.transpose((0, 2, 3, 1)).view(1, int(size_h * size_w),
                                                                     int(num_classes))

            prev_hm = []
            prev_track_id = []
            class_id = 0
            for task in cfg.tasks:
                prev_hm.append(prev_hm_[..., class_id: class_id + task['num_class']])
                prev_track_id.append(prev_track_id_[..., class_id: class_id + task['num_class']])
                class_id += task['num_class']

            preds = ms_model(data_batch, return_loss=False,
                          return_feature=True)  # jiance  data_batch: 'metadata' points, voxels. shape, num_points, num_voxels, coordinates, ref_from_car, car_from_global
            # pre: 6个reg,每个reg里面有reg,height,dim,rot,vel,hm  prev_hm: 6个list, prev_track_id：6个list
            outputs, track_outputs = ms_model.bbox_head.predict_tracking(data_batch, preds, ms_model.test,
                                                                      prev_hm=prev_hm, prev_track_id=prev_track_id,
                                                                      new_only=False)
            outputs[0]['tracking_id'] = mindspore.numpy.arange(start_id, start_id + outputs[0]['scores'].shape[0]).astype('int')
            start_id += outputs[0]['scores'].size(0)

        else:  # first frame
            ms_model.set_train(False)
            # torch_model.eval()
            outputs = ms_model(data_batch, return_loss=False)


            # data_batch["voxels"] = torch.Tensor(data_batch["voxels"].asnumpy())
            # data_batch["coordinates"] = torch.Tensor(data_batch["coordinates"].asnumpy())
            # data_batch["num_points"] = torch.Tensor(data_batch["num_points"].asnumpy())
            # data_batch["num_voxels"] = torch.Tensor(data_batch["num_voxels"].asnumpy())
            # data_batch["shape"] = torch.Tensor(data_batch["shape"].asnumpy())
            # outputs1 = torch_model(data_batch, return_loss=False)

            # outputs[0]['box3d_lidar'] = torch.tensor(outputs[0]['box3d_lidar'].asnumpy())
            # outputs[0]['scores'] = torch.tensor(outputs[0]['scores'].asnumpy())
            # outputs[0]['label_preds'] = torch.tensor(outputs[0]['label_preds'].asnumpy())
            # outputs[0]['token'] = torch.tensor(outputs[0]['token'])


            outputs[0]['tracking_id'] = mindspore.numpy.arange(start_id, start_id + outputs[0]['scores'].shape[0]).astype('int')
            start_id += outputs[0]['scores'].shape[0]

        output = outputs[0].copy()
        token = output["token"]
        cat = ops.Concat(axis=0)
        for k, v in output.items():
            if k not in ["token"]:
                if track_outputs is not None:
                    output[k] = cat([v.copy(), track_outputs[0][k].copy()])

        detections.update({token: output, })

        prev_output = {}
        box3d_lidar = output['box3d_lidar'].copy().asnumpy()
        box3d = np.concatenate((box3d_lidar[:, :3], np.ones((box3d_lidar.shape[0], 1))), axis=1).T
        box3d = (np.linalg.inv(data_batch['car_from_global'][0].asnumpy()) @ np.linalg.inv(
            data_batch['ref_from_car'][0].asnumpy())) @ box3d
        prev_output['box3d_lidar'] = box3d_lidar
        prev_output['box3d_global'] = box3d
        prev_output['label_preds'] = output['label_preds'].asnumpy()
        prev_output['scores'] = output['scores'].asnumpy()
        prev_output['tracking_id'] = output['tracking_id'].asnumpy()
        prev_detections[str(output['token'])] = prev_output

    synchronize()

    all_predictions = all_gather(detections)

    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    # args.eval_det = True
    if args.eval_det:
        result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=args.work_dir, testset=False)
        if result_dict is not None:
            for k, v in result_dict["results"].items():
                print(f"Evaluation {k}: {v}")

    # eval tracking
    dataset.evaluation_tracking(copy.deepcopy(predictions), output_dir=args.work_dir, testset=False)


def render_trackmap(preds_dicts, grids, cfg):
    prev_hm = np.zeros((1, num_classes, size_h, size_w), dtype=np.float32)
    prev_tracking_map = np.zeros((1, num_classes, size_h, size_w), dtype=np.int64) - 1
    label_preds = preds_dicts['label_preds']
    box3d_lidar = preds_dicts['box3d_lidar']
    scores = preds_dicts['scores']
    tracking_ids = preds_dicts['tracking_id']

    box_corners = center_to_corner_box2d(box3d_lidar[:, :2], box3d_lidar[:, 3:5], box3d_lidar[:, -1])
    box_corners = (box_corners - voxel_range[:2].reshape(1, 1, 2)) / voxel_size[:2].reshape(1, 1, 2) / downsample
    masks = points_in_convex_polygon_jit(grids, box_corners)

    for obj in range(label_preds.shape[0]):
        cls_id = label_preds[obj]
        score = scores[obj]
        tracking_id = tracking_ids[obj]
        size_x, size_y = box3d_lidar[obj, 3] / voxel_size[0] / downsample, box3d_lidar[obj, 4] / voxel_size[
            1] / downsample
        if size_x > 0 and size_y > 0:
            radius = gaussian_radius((size_y, size_x), min_overlap=0.1)
            radius = min(cfg.assigner.min_radius, int(radius))

            coor_x = (box3d_lidar[obj, 0] - voxel_range[0]) / voxel_size[0] / downsample
            coor_y = (box3d_lidar[obj, 1] - voxel_range[1]) / voxel_size[1] / downsample
            ct = np.array([coor_x, coor_y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            # throw out not in range objects to avoid out of array area when creating the heatmap
            if not (0 <= ct_int[0] < size_w and 0 <= ct_int[1] < size_h):
                continue
                # render center map as in centertrack
            draw_gaussian(prev_hm[0, cls_id], ct, radius, score)  #

            # tracking ID map
            mask = masks[:, obj].nonzero()[0]
            coord_in_box = grids[mask, :]
            mask1 = prev_tracking_map[0, cls_id][coord_in_box[:, 1], coord_in_box[:, 0]] == -1
            mask2 = prev_hm[0, cls_id][coord_in_box[:, 1], coord_in_box[:, 0]] < score
            mask = mask[np.logical_or(mask1, mask2)]
            coord_in_box = grids[mask, :]
            prev_tracking_map[0, cls_id][coord_in_box[:, 1], coord_in_box[:, 0]] = tracking_id
            prev_tracking_map[0, cls_id][ct_int[1], ct_int[0]] = tracking_id

    return Tensor(prev_hm, mindspore.float32), Tensor(prev_tracking_map, mindspore.float32)


def meshgrid(w, h):
    ww, hh = np.meshgrid(range(w), range(h))
    ww = ww.reshape(-1)
    hh = hh.reshape(-1)

    return np.stack([ww, hh], axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="train config file path",
                        default='/home/zad/project/simtrack/examples/point_pillars/configs/nusc_all_pp_centernet_tracking.py')
    parser.add_argument("--work_dir", help="the dir to save logs and models",
                        default='/home/zad/project/simtrack/word_dirs/baseline')
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from",
        default='/home/zad/project/ms_sim/simtrack_pillar.ckpt')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--eval_det", action='store_true')

    args = parser.parse_args()
    tracking(args)
    # cfg = Config.fromfile(args.config)
    # dataset = build_dataset(cfg.data.val)
    # p = dataset.__getitem__(20)
    # print(p)

