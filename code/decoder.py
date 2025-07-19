import torch
import torch.nn as nn
import torch.nn.functional as F


class upConv(nn.Module):
    def __init__(self, input_size=(512, 32, 64)):
        super(upConv, self).__init__()
        self.input_size = input_size

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=False),
            )

        self.decoder = nn.Sequential(
            # up_block(256, 256),  # 512 -> 256 채널
            # up_block(256, 128),  # 256 -> 128 채널
            # up_block(128, 64),  # 128 -> 64 채널
            up_block(16, 64),
            up_block(64, 64),
            up_block(64, 32),  # 64 -> 32 채널
            up_block(32, 32),  # 64 -> 32 채널
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        device = x.device

        if x.ndimension() == 4 and x.shape[1:] == self.input_size:
            estimated_img = self.decoder(x)
            return estimated_img

        raise ValueError(
            f"Unexpected input shape: {x.shape}. Expected (B, {self.input_size})."
        )
