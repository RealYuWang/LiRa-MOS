import torch
import torch.nn as nn

def fuse_enc_dec(enc, dec):
    new_dec = dec
    diff = enc.shape[2] - dec.shape[2]
    if diff > 0:
        new_dec = nn.functional.pad(dec, (0, diff), mode='constant', value=0)
    return torch.cat([new_dec, enc], dim=1)

class RadarMOSNet(nn.Module):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super(RadarMOSNet, self).__init__()
        self.model_cfg = model_cfg
        # Encoder
        self.enc1 = self._encoder_block(num_point_features, 64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.enc2 = self._encoder_block(64, 128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.enc3 = self._encoder_block(128, 256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck = self._encoder_block(256, 512)

        # Decoder
        self.upconv3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._decoder_block(512, 256)
        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._decoder_block(256, 128)
        self.upconv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._decoder_block(128, 64)

        self.final_conv = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, batch_dict):
        x = batch_dict['points'][:,1:] # 去掉batch_idx
        # print(f'Number of points: {x.shape[0]}')
        x = x.permute(1, 0).unsqueeze(0) # (N, 7) -> (1, 7, N)
        # batch_idx = batch_dict['points'][:,0]

        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)

        bottleneck = self.bottleneck(pool3) # (B, 512, N/8)

        # Decoder with skip connections
        upconv3 = self.upconv3(bottleneck)
        merged3 = fuse_enc_dec(enc3, upconv3)
        dec3 = self.dec3(merged3)
        upconv2 = self.upconv2(dec3)
        merged2 = fuse_enc_dec(enc2, upconv2)
        dec2 = self.dec2(merged2)
        upconv1 = self.upconv1(dec2)
        merged1 = fuse_enc_dec(enc1, upconv1)
        dec1 = self.dec1(merged1)

        output = torch.sigmoid(self.final_conv(dec1)) # (1, 1, N)
        batch_dict.update({
            'mos_pred': output.permute(2, 1, 0).squeeze(), # (N,)
            'mos_feature': bottleneck
        })
        return batch_dict

    def _encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def get_loss(self, batch_dict):
        loss_func = nn.BCELoss()
        mos_pred = batch_dict['mos_pred']
        mos_gt = batch_dict['mos_label']
        return loss_func(mos_pred, mos_gt)
