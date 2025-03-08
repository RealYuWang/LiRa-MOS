import warnings
warnings.simplefilter("ignore")
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.datasets import build_dataloader
import kornia
import numpy as np


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if key == 'camera_imgs':
            batch_dict[key] = val.cuda()
        elif not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'image_paths','ori_shape','img_process_infos']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()

model_cfg_file = '/home/yu/OpenPCDet/tools/cfgs/kitti_models/RLNet.yaml'
cfg_from_yaml_file(model_cfg_file, cfg) # 自动加载了数据集配置文件到DATA_CONFIG下

train_set, train_loader, _ = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=8,
    dist=False, workers=4,
    logger=None,
    training=True
)

# collate_batch 为点云添加了batch index维度
batch = next(iter(train_loader))
print(batch.keys())
print(batch['frame_id'])
print('batch radar points: ',batch['points'].shape)
print('batch lidar points: ',batch['points_lidar'].shape)
print('batch mos label: ', batch['mos_label'].shape)
load_data_to_gpu(batch)

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
model.cuda()
model.train()

# ret_dict: loss, tb_dict: tensor board, disp_dict： 调试信息
ret_dict, tb_dict, disp_dict = model(batch)
print(ret_dict)
print(tb_dict)

