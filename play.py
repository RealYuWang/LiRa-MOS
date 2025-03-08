import numpy as np
from pcdet.utils.common_utils import remove_ground_points

points = np.fromfile('/datasets/vod/lidar/training/velodyne/00010.bin', dtype=np.float32).reshape(-1, 4)
print(points.shape)
left = remove_ground_points(points, 0.05)
print(left.shape)