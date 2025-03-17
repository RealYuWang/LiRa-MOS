import warnings
warnings.simplefilter("ignore")
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
import numpy as np

model_cfg_file = 'cfgs/kitti_models/RLNet.yaml'
cfg_from_yaml_file(model_cfg_file, cfg) # 自动加载了数据集配置文件到DATA_CONFIG下

train_set, train_loader, _ = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=8,
    dist=False, workers=8,
    logger=None,
    training=True
)

batch = next(iter(train_loader))
# points = batch['points']
# batch_indices = np.unique(points[:, 0])
# for batch_idx in batch_indices:
#     batch_points = points[points[:, 0] == batch_idx]  # 过滤出当前 batch 的点
#     print(f"Batch {int(batch_idx)}: {batch_points.shape[0]} points")

load_data_to_gpu(batch)

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
model.cuda()
model.train()

ret_dict, tb_dict, disp_dict = model(batch)
print(ret_dict)
print(tb_dict)



