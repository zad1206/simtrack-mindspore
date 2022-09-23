import copy

from ms_sim.utils.registry import build_from_cfg
from ms_sim.datasets.dataset_wrappers import ConcatDataset, RepeatDataset
from ms_sim.datasets.registry import DATASETS

def build_dataset(cfg, default_args=None):
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg["type"] == "RepeatDataset":
        dataset = RepeatDataset(
            build_dataset(cfg["dataset"], default_args), cfg["times"]
        )
    # elif isinstance(cfg['ann_file'], (list, tuple)):
    #     dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset

if __name__ == "__main__":
    from simtrack.det3d.torchie import Config
    config = '/home/zad/project/simtrack/examples/point_pillars/configs/nusc_all_pp_centernet_tracking.py'
    cfg = Config.fromfile(config)
    dataset = build_dataset(cfg.data.val)