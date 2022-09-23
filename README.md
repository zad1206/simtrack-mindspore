# simtrack-mindspore
Implemented the simtrack model based on mindspore
## Exploring Simple 3D Multi-Object Tracking for Autonomous Driving

[[Paper]](https://arxiv.org/pdf/2108.10312.pdf) 

## requirements
```
Ubuntu 20.04
python=3.8
mindspore=1.8
cuda=11.1
```

## Getting Started

### Data Preparation 
* [nuScenes](https://www.nuscenes.org)
```
python ./datasets/create_data.py nuscenes_data_prep --root_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --nsweeps=10
```

### Training
```
python -m torch.distributed.launch --nproc_per_node=8 ./train.py examples/point_pillars/configs/nusc_all_pp_centernet_tracking.py --work_dir SAVE_DIR
```

### Test
In `./model_zoo` we provide our trained (pillar based) model on nuScenes.          
Note: We currently only support inference with a single GPU.
```
python ./val_nusc_tracking.py examples/point_pillars/configs/nusc_all_pp_centernet_tracking.py --checkpoint CHECKPOINTFILE  --work_dir SAVE_DIR
```
