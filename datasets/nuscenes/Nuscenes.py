import sys
import pickle
import json
import random
import operator
import numpy as np
from pathlib import Path

from nuscenes.nuscenes import NuScenes
from ms_sim.datasets.custom import PointCloudDataset
from ms_sim.datasets.nuscenes.nusc_commom import (
    general_to_detection,
    cls_attr_dist,
    _second_det_to_nusc_box,
    _lidar_nusc_box_to_global,
    eval_main
)
from ms_sim.datasets.registry import DATASETS

data_keys = ['metadata', 'points', 'voxels', 'shape', 'num_points', 'num_voxels', 'ref_from_car', 'car_from_global', "coordinates"]


@DATASETS.register_module
class NuScenesDataset(PointCloudDataset):
    NumPointFeatures = 5  # x, y, z, intensity, timestamp, (beam_id)

    def __init__(
            self,
            info_path,
            root_path,
            nsweeps=1,
            cfg=None,
            pipeline=None,
            class_names=None,
            test_mode=False,
            version="v1.0-mini",
            **kwargs,
    ):
        super(NuScenesDataset, self).__init__(
             root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names)

        self.nsweeps = nsweeps
        assert self.nsweeps > 0, "At least input one sweep please!"

        self._info_path = info_path
        self._class_names = class_names
        self.data_keys = data_keys

        if not hasattr(self, "_nusc_infos"):
            self.load_infos(self._info_path)

        self._num_point_features = NuScenesDataset.NumPointFeatures
        self._name_mapping = general_to_detection

        self.version = version
        self.eval_version = "detection_cvpr_2019"

    def reset(self):
        self.logger.info(f"re-sample {self.frac} frames from full set")
        random.shuffle(self._nusc_infos_all)
        self._nusc_infos = self._nusc_infos_all[: self.frac]

    def load_infos(self, info_path):

        with open(self._info_path, "rb") as f:
            _nusc_infos_all = pickle.load(f)

        if not self.test_mode:
            self.frac = int(len(_nusc_infos_all) * 0.25)

            _cls_infos = {name: [] for name in self._class_names}
            for info in _nusc_infos_all:
                for name in set(info["gt_names"]):
                    if name in self._class_names:
                        _cls_infos[name].append(info)

            duplicated_samples = sum([len(v) for _, v in _cls_infos.items()])
            _cls_dist = {k: len(v) / duplicated_samples for k, v in _cls_infos.items()}

            self._nusc_infos = []

            frac = 1.0 / len(self._class_names)
            ratios = [frac / v for v in _cls_dist.values()]

            for cls_infos, ratio in zip(list(_cls_infos.values()), ratios):
                self._nusc_infos += np.random.choice(
                    cls_infos, int(len(cls_infos) * ratio)
                ).tolist()

            _cls_infos = {name: [] for name in self._class_names}
            for info in self._nusc_infos:
                for name in set(info["gt_names"]):
                    if name in self._class_names:
                        _cls_infos[name].append(info)

            _cls_dist = {
                k: len(v) / len(self._nusc_infos) for k, v in _cls_infos.items()
            }
        else:
            if isinstance(_nusc_infos_all, dict):
                self._nusc_infos = []
                for v in _nusc_infos_all.values():
                    self._nusc_infos.extend(v)
            else:
                self._nusc_infos = _nusc_infos_all

    def __len__(self):

        if not hasattr(self, "_nusc_infos"):
            self.load_infos(self._info_path)

        return len(self._nusc_infos)

    @property
    def ground_truth_annotations(self):
        if "gt_boxes" not in self._nusc_infos[0]:
            return None
        gt_annos = []
        for info in self._nusc_infos:
            gt_annos.append({"token": info["token"]})
        return gt_annos


    def __getitem__(self, idx):
        info = self._nusc_infos[idx]
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "nsweeps": self.nsweeps,
                "annotations": None,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": info["token"],
            },
            'ref_from_car': info['ref_from_car'],
            'car_from_global': info['car_from_global'],
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
        }
        data, _ = self.pipeline(res, info)
        token = data['metadata']['token']
        points = list(data['points'])
        voxels = list(data['voxels'])
        shape = list(data['shape'])
        num_points = list(data['num_points'])
        num_voxels = list(data['num_voxels'])
        coordinates = list(data['coordinates'])
        ref_from_car = list(data['ref_from_car'])
        car_from_global = list(data['car_from_global'])
        return token, points, voxels, shape, num_points, num_voxels, coordinates, ref_from_car, car_from_global

    def evaluation(self, detections, output_dir=None, testset=False):
        version = self.version
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
            "v1.0-test": "test",
        }

        if not testset:
            dets = []
            gt_annos = self.ground_truth_annotations
            assert gt_annos is not None

            miss = 0
            for gt in gt_annos:
                try:
                    dets.append(detections[gt["token"]])
                except Exception:
                    miss += 1

            assert miss == 0
        else:
            dets = [v for _, v in detections.items()]

        nusc_annos = {
            "results": {},
            "meta": None,
        }

        nusc = NuScenes(version=version, dataroot=str(self._root_path), verbose=True)

        mapped_class_names = []
        for n in self._class_names:
            if n in self._name_mapping:
                mapped_class_names.append(self._name_mapping[n])
            else:
                mapped_class_names.append(n)

        for det in dets:
            annos = []
            boxes = _second_det_to_nusc_box(det)
            boxes = _lidar_nusc_box_to_global(nusc, boxes, det["metadata"]["token"])
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = None
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = None

                nusc_anno = {
                    "sample_token": det["metadata"]["token"],
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": attr
                    if attr is not None
                    else max(cls_attr_dist[name].items(), key=operator.itemgetter(1))[
                        0
                    ],
                }
                annos.append(nusc_anno)
            nusc_annos["results"].update({det["metadata"]["token"]: annos})

        nusc_annos["meta"] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        name = self._info_path.split("/")[-1].split(".")[0]
        res_path = str(Path(output_dir) / Path(name + ".json"))
        with open(res_path, "w") as f:
            json.dump(nusc_annos, f)

        print(f"Finish generate predictions for testset, save to {res_path}")

        if not testset:
            eval_main(
                nusc,
                self.eval_version,
                res_path,
                eval_set_map[self.version],
                output_dir,
            )

            with open(Path(output_dir) / "metrics_summary.json", "r") as f:
                metrics = json.load(f)

            detail = {}
            result = f"Nusc {version} Evaluation\n"
            for name in mapped_class_names:
                detail[name] = {}
                for k, v in metrics["label_aps"][name].items():
                    detail[name][f"dist@{k}"] = v
                threshs = ", ".join(list(metrics["label_aps"][name].keys()))
                scores = list(metrics["label_aps"][name].values())
                mean = sum(scores) / len(scores)
                scores = ", ".join([f"{s * 100:.2f}" for s in scores])
                result += f"{name} Nusc dist AP@{threshs}\n"
                result += scores
                result += f" mean AP: {mean}"
                result += "\n"
            res_nusc = {
                "results": {"nusc": result},
                "detail": {"nusc": detail},
            }
        else:
            res_nusc = None

        if res_nusc is not None:
            res = {
                "results": {"nusc": res_nusc["results"]["nusc"], },
                "detail": {"eval.nusc": res_nusc["detail"]["nusc"], },
            }
        else:
            res = None

        return res, None

    def evaluation_tracking(self, detections, output_dir=None, testset=False):

        version = self.version
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
            "v1.0-test": "test",
        }

        if not testset:
            dets = []
            gt_annos = self.ground_truth_annotations
            assert gt_annos is not None

            miss = 0
            for gt in gt_annos:
                try:
                    dets.append(detections[gt["token"]])
                except Exception:
                    miss += 1

            assert miss == 0
        else:
            dets = [v for _, v in detections.items()]

        nusc_annos = {
            "results": {},
            "meta": None,
        }

        nusc = NuScenes(version=version, dataroot=str(self._root_path), verbose=True)

        mapped_class_names = []
        for n in self._class_names:
            if n in self._name_mapping:
                mapped_class_names.append(self._name_mapping[n])
            else:
                mapped_class_names.append(n)

        for det in dets:
            annos = []
            boxes = _second_det_to_nusc_box(det)
            boxes = _lidar_nusc_box_to_global(nusc, boxes, det["metadata"]["token"])
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if name in ['construction_vehicle', "barrier", "traffic_cone"]:
                    continue
                nusc_anno = {
                    "sample_token": det["metadata"]["token"],
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "tracking_name": name,
                    "tracking_score": box.score,
                    "tracking_id": str(int(det['tracking_id'][i])),
                }
                annos.append(nusc_anno)
            nusc_annos["results"].update({det["metadata"]["token"]: annos})

        nusc_annos["meta"] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        name = self._info_path.split("/")[-1].split(".")[0]
        res_path = str(Path(output_dir) / Path('tracking_results' + ".json"))
        with open(res_path, "w") as f:
            json.dump(nusc_annos, f)

        print(f"Finish generate predictions for testset, save to {res_path}")

        from nuscenes.eval.tracking.evaluate import TrackingEval
        from nuscenes.eval.common.config import config_factory as track_configs

        cfg = track_configs("tracking_nips_2019")
        nusc_eval = TrackingEval(
            config=cfg,
            result_path=res_path,
            eval_set='mini_val',
            output_dir=output_dir,
            verbose=True,
            nusc_version="v1.0-mini",
            nusc_dataroot=self._root_path,
        )
        metrics_summary = nusc_eval.main()

if __name__ == '__main__':
    info_path = '/home/zad/project/v1.0-mini/infos_train_10sweeps_tracking.pkl'
    root_path = '/home/zad/project/v1.0-mini'
    pipeline = [{'type': 'LoadPointCloudFromFile', 'dataset': 'NuScenesDataset', 'nsweeps': 10},
                {'type': 'LoadPointCloudAnnotations', 'with_bbox': True},
                {'type': 'Preprocess',
                 'cfg': {'mode': 'train', 'shuffle_points': True, 'global_rot_noise': [-0.3925, 0.3925],
                         'global_scale_noise': [0.95, 1.05], 'global_trans_noise': [0.2, 0.2, 0.2],
                         'remove_points_after_sample': False, 'remove_unknown_examples': False, 'min_points_in_gt': 0,
                         'flip': [0.5, 0.5], 'db_sampler': {'type': 'GT-AUG', 'enable': True,
                                                            'db_info_path': '/home/zad/project/v1.0-mini/dbinfos_train_1sweeps.pkl',
                                                            'sample_groups': [{'car': 2}, {'truck': 3},
                                                                              {'construction_vehicle': 7}, {'bus': 4},
                                                                              {'trailer': 6}, {'motorcycle': 2},
                                                                              {'bicycle': 6}, {'pedestrian': 2}],
                                                            'db_prep_steps': [{'filter_by_min_num_points': {'car': 5,
                                                                                                            'truck': 5,
                                                                                                            'bus': 5,
                                                                                                            'trailer': 5,
                                                                                                            'construction_vehicle': 5,
                                                                                                            'traffic_cone': 5,
                                                                                                            'barrier': 5,
                                                                                                            'bicycle': 5,
                                                                                                            'motorcycle': 5,
                                                                                                            'pedestrian': 5}},
                                                                              {'filter_by_difficulty': [-1]}],
                                                            'rate': 1.0, 'gt_drop_percentage': 0.5,
                                                            'gt_drop_max_keep_points': 5, 'point_dim': 5},
                         'class_names': ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                                         'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']}},
                {'type': 'Voxelization',
                 'cfg': {'range': (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0), 'voxel_size': (0.2, 0.2, 8),
                         'max_points_in_voxel': 20, 'max_voxel_num': 30000}},
                {'type': 'AssignTracking', 'cfg': {'target_assigner': {
                    'tasks': [{'num_class': 1, 'class_names': ['car'], 'stride': 1},
                              {'num_class': 2, 'class_names': ['truck', 'construction_vehicle'], 'stride': 1},
                              {'num_class': 2, 'class_names': ['bus', 'trailer'], 'stride': 1},
                              {'num_class': 1, 'class_names': ['barrier'], 'stride': 1},
                              {'num_class': 2, 'class_names': ['motorcycle', 'bicycle'], 'stride': 1},
                              {'num_class': 2, 'class_names': ['pedestrian', 'traffic_cone'], 'stride': 1}]},
                                                   'out_size_factor': 4, 'gaussian_overlap': 0.1, 'max_objs': 500,
                                                   'min_radius': 2}},
                {'type': 'Reformat'}]
    class_name = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle',
                  'pedestrian', 'traffic_cone']
    kwargs = {'ann_file': '/home/zad/project/v1.0-mini/infos_train_10sweeps_tracking.pkl', 'n_sweeps': 10}
    data = NuScenesDataset(info_path=info_path, root_path=root_path, pipeline=pipeline, class_names=class_name)
    print(data)
    dataset = data.batch(1)
    # print(p)