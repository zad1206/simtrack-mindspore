import os.path as osp
from pathlib import Path

import numpy as np

from ms_sim.datasets.registry import DATASETS
from ms_sim.datasets.pipelines.Compose import Compose


@DATASETS.register_module
class PointCloudDataset:
    """An abstract class.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    NumPointFeatures = -1
    CLASSES = None

    def __init__(
            self,
            root_path,
            info_path,
            pipeline=None,
            test_mode=False,
            class_names=None,
            **kwrags
    ):
        self._info_path = info_path
        self._root_path = Path(root_path)
        self._class_names = class_names

        self.test_mode = test_mode

        self._set_group_flag()

        if pipeline is None:
            self.pipeline = None
        else:
            self.pipeline = Compose(pipeline)
            a = 1

    def __getitem__(self, index):
        """This function is used for preprocess.
        you need to create a input dict in this function for network inference.
        format: {
            anchors
            voxels
            num_points
            coordinates
            if training:
                labels
                reg_targets
            [optional]anchors_mask, slow in SECOND v1.5, don't use this.
            [optional]metadata,
        }
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_sensor_data(self, query):
        """Dataset must provide a unified function to get data.
        Args:
            query: int or dict. this param must support int for training.
                if dict, should have this format (no example yet):
                {
                    sensor_name: {
                        sensor_meta
                    }
                }
                if int, will return all sensor data.
                (TODO: how to deal with unsynchronized data?)
        Returns:
            sensor_data: dict.
            if query is int (return all), return a dict with all sensors:
            {
                sensor_name: sensor_data
                ...
                metadata: ...
            }

            if sensor is lidar (all lidar point cloud must be concatenated to one array):
            e.g. If your dataset have two lidar sensor, you need to return a single dict:
            {
                "lidar": {
                    "points": ...
                    ...
                }
            }
            sensor_data: {
                points: [N, 3+]
                [optional]annotations: {
                    "boxes": [N, 7] locs, dims, yaw, in lidar coord system. must tested
                        in provided visualization tools such as second.utils.simplevis
                        or web tool.
                    "names": array of string.
                }
            }
            if sensor is camera (not used yet):
            sensor_data: {
                data: image string (array is too large)
                [optional]annotations: {
                    "boxes": [N, 4] 2d bbox
                    "names": array of string.
                }
            }
            metadata: {
                # dataset-specific information.
                image_idx: ...
            }
            [optional]calib
        """
        raise NotImplementedError

    def evaluation(self, dt_annos, output_dir):
        """Dataset must provide a evaluation function to evaluate model."""
        raise NotImplementedError

    @property
    def ground_truth_annotations(self):
        """
        ground_truth_annotations format:
        {
            bbox: [N, 4], if you fill fake data, MUST HAVE >25 HEIGHT!!!!!!
            alpha: [N], you can use -10 to ignore it.
            occluded: [N], you can use zero.
            truncated: [N], you can use zero.
            name: [N]
            location: [N, 3] center of 3d box.
            dimensions: [N, 3] dim of 3d box.
            rotation_y: [N] angle.
        }
        all fields must be filled, but some fields can fill
        zero.
        """
        raise NotImplementedError

    def pre_pipeline(self, results):
        raise NotImplementedError

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.ones(len(self), dtype=np.uint8)
        # self.flag = np.zeros(len(self), dtype=np.uint8)
        # for i in range(len(self)):
        #     img_info = self.img_infos[i]
        #     if img_info['width'] / img_info['height'] > 1:
        #         self.flag[i] = 1

    def prepare_train_input(self, idx):
        raise NotImplementedError

        # img_info = self.img_infos[idx]
        # ann_info = self.get_ann_info(idx)
        # results = dict(img_info=img_info, ann_info=ann_info)
        # if self.proposals is not None:
        #     results['proposals'] = self.proposals[idx]
        # self.pre_pipeline(results)
        # return self.pipeline(results)

    def prepare_test_input(self, idx):
        raise NotImplementedError

        # img_info = self.img_infos[idx]
        # results = dict(img_info=img_info)
        # if self.proposals is not None:
        #     results['proposals'] = self.proposals[idx]
        # self.pre_pipeline(results)
        # return self.pipeline(results)