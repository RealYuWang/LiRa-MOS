import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
import random
import numpy as np

class SEAttention(nn.Module):
    def __init__(self, channel,ratio = 16):
        super(SEAttention, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
                nn.Linear(in_features=channel, out_features=channel // ratio),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=channel // ratio, out_features=channel),
                nn.Sigmoid()
            )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)

class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        if self.model_cfg.WEIGHT:
            self.weight_conv = ConvModule(
                    2,
                    2,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01))

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        # 来自 VoxelResBackBone8x
        if batch_dict.get('encoded_spconv_tensor', None) is not None:
            encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
            spatial_features = encoded_spconv_tensor.dense()
            N, C, D, H, W = spatial_features.shape
            spatial_features = spatial_features.view(N, C * D, H, W)

        if batch_dict.get('encoded_spconv_tensor_lidar', None) is not None:
            encoded_spconv_tensor_lidar = batch_dict['encoded_spconv_tensor_lidar']
            spatial_features_lidar = encoded_spconv_tensor_lidar.dense()
            N, C, D, H, W = spatial_features_lidar.shape
            spatial_features_lidar = spatial_features_lidar.view(N, C * D, H, W)

        if hasattr(self, 'weight_conv'):
            # if hasattr(self, 'weight_conv'):
            if False:
                concat_feat = torch.cat([torch.mul(w1, spatial_features_lidar), torch.mul(w2, spatial_features)], dim=1)
                batch_dict['spatial_features'] = concat_feat
            else:
                if self.training:
                    rand_num = random.random()
                    if rand_num > 0.8:
                        rand_drop = random.random()
                        if rand_drop > 0.8:
                            spatial_features_lidar = spatial_features_lidar * torch.zeros_like(spatial_features_lidar)
                        else:
                            spatial_features = spatial_features * torch.zeros_like(spatial_features)
                x_mean = torch.mean(spatial_features_lidar, dim=1, keepdim=True)
                r_x_mean = torch.mean(spatial_features, dim=1, keepdim=True)
                mix_x = torch.cat([x_mean, r_x_mean], dim=1)
                mix_x = self.weight_conv(mix_x)
                weight = torch.nn.functional.softmax(mix_x, dim=1)
                w1 = torch.split(weight, 1, dim=1)[0]
                w2 = torch.split(weight, 1, dim=1)[1]
                # batch_dict['RLNet-Lidar'] = spatial_features_lidar
                # batch_dict['RLNet-Radar'] = spatial_features
                # batch_dict['RLNet-Fusion-L'] = torch.mul(w1, spatial_features_lidar)
                # batch_dict['RLNet-Fusion-R'] = torch.mul(w1, spatial_features)

                batch_dict['spatial_features'] = torch.cat(
                    [torch.mul(w1, spatial_features_lidar), torch.mul(w2, spatial_features)], dim=1)
                batch_dict['weight_radar'] = w1
                # batch_dict['spatial_features'] = torch.cat([spatial_features_lidar, spatial_features], dim=1)
        else:

            # batch_dict['spatial_features'] = torch.cat([spatial_features_proj_L, spatial_features],dim=1)
            if batch_dict.get('encoded_spconv_tensor', None) is not None:
                # mid_features = self.decoder(spatial_features)
                # extra_features = batch_dict['extra_features']
                # coor_features = batch_dict['coor_features']
                # # num_features = batch_dict['num_features']
                # batch_dict['spatial_features'] = torch.cat([spatial_features, coor_features], dim=1)
                # batch_dict['spatial_features'] = extra_features
                batch_dict['spatial_features'] = spatial_features
            else:
                batch_dict['spatial_features'] = spatial_features_lidar


        # batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride_lidar']
        return batch_dict
