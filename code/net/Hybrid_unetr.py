import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet1d import resnet1d34
from .resnet2d import resnet2d34
from .tr import SelfCrossTR


class DoubleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, groups=32):
        super(DoubleConv1d, self).__init__()
        ng = min(groups, out_channels)
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(ng, out_channels),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(ng, out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=32):
        super(DoubleConv, self).__init__()
        ng = min(groups, out_channels)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(ng, out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(ng, out_channels),
            nn.ReLU(inplace=False),
            nn.Dropout2d(0.0)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


#------------- upper part is classic U-Net module -------------#


class Hybrid_unetr(nn.Module):
    def __init__(self, bilinear=True):
        super(Hybrid_unetr, self).__init__()
        
        self.conv1_1d = DoubleConv1d(4, 64)
        self.scale = nn.Parameter(torch.tensor(4.0))
        self.bias = nn.Parameter(torch.tensor(10.0))
        self.vision_encoder = resnet2d34()
        self.rir_encoder = resnet1d34()
  
        self.TR1 = SelfCrossTR(
            n_head=8,
            n_blocks=4,
            img_in_channels=16,
            rir_in_channels=64,
            d_model=512,
            img_size=(256,512),
            img_patch_size=16,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.TR2 = SelfCrossTR(
            n_head=8,
            n_blocks=4,
            img_in_channels=32,
            rir_in_channels=128,
            d_model=512,
            img_size=(128,256),
            img_patch_size=8,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.TR3 = SelfCrossTR(
            n_head=8,
            n_blocks=4,
            img_in_channels=64,
            rir_in_channels=256,
            d_model=512,
            img_size=(64,128),
            img_patch_size=4,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.TR4 = SelfCrossTR(
            n_head=8,
            n_blocks=4,
            img_in_channels=128,
            rir_in_channels=512,
            d_model=512,
            img_size=(32,64),
            img_patch_size=2,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.TR5 = SelfCrossTR(
            n_head=8,
            n_blocks=4,
            img_in_channels=256,
            rir_in_channels=1024,
            d_model=512,
            img_size=(16,32),
            img_patch_size=1,
            dim_feedforward=2048,
            dropout=0.1
        )
        
        self.up1 = Up(256, 128, 128, bilinear)  
        self.up2 = Up(128, 64, 64, bilinear)     
        self.up3 = Up(64, 32, 32, bilinear)    
        self.up4 = Up(32, 16, 16, bilinear)                 

        self.outc = OutConv(16, 1)
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU(inplace=False)
        self.gn1_vision = nn.GroupNorm(num_groups=1, num_channels=16)
        self.gn1_rir = nn.GroupNorm(num_groups=4, num_channels=64)

    def _scale_2d(self, x, eps=1e-5):
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)
        return (x - mean) / (std + eps)

    def _rescale_2d(self, x_scaled, original, eps=1e-5):
        mean = original.mean(dim=[2, 3], keepdim=True)
        std = original.std(dim=[2, 3], keepdim=True)
        return x_scaled * (std + eps) + mean

    def _scale_1d(self, x, eps=1e-5):
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        return (x - mean) / (std + eps)

    def _rescale_1d(self, x_scaled, original, eps=1e-5):
        mean = original.mean(dim=2, keepdim=True)
        std = original.std(dim=2, keepdim=True)
        return x_scaled * (std + eps) + mean
        
    def forward(self, img, rir, mic_info, mask_pos):    # mask pos scale: 512 x 1024
        
        img1 = self.vision_encoder.conv1(img)    
        img1 = self.gn1_vision(img1)
        img1 = self.relu(img1)  
        img2 = self.vision_encoder.maxpool(img1)
        img2 = self.vision_encoder.layer1(img2)   
        img3 = self.vision_encoder.layer2(img2)
        img4 = self.vision_encoder.layer3(img3)  
        img5 = self.vision_encoder.layer4(img4)  
        
        rir1 = self.conv1_1d(rir)  
        rir1 = self.gn1_rir(rir1)
        rir1 = self.relu(rir1)
        rir2 = self.rir_encoder.maxpool(rir1)
        rir2 = self.rir_encoder.layer1(rir2)   
        rir3 = self.rir_encoder.layer2(rir2)
        rir4 = self.rir_encoder.layer3(rir3)  
        rir5 = self.rir_encoder.layer4(rir4)
        
        # print("img1 max/min:", img1.max().item(), img1.min().item())
        # print("img2 max/min:", img2.max().item(), img2.min().item())
        # print("img3 max/min:", img3.max().item(), img3.min().item())
        # print("img4 max/min:", img4.max().item(), img4.min().item())
        # print("img5 max/min:", img5.max().item(), img5.min().item())
        
        # print("rir1 max/min:", rir1.max().item(), rir1.min().item())
        # print("rir2 max/min:", rir2.max().item(), rir2.min().item())
        # print("rir3 max/min:", rir3.max().item(), rir3.min().item())
        # print("rir4 max/min:", rir4.max().item(), rir4.min().item())
        # print("rir5 max/min:", rir5.max().item(), rir5.min().item())
        
        # Vision encoder + scale
        (tr1, rir_out1) = self.TR1(img1, rir1, mic_info, mask_pos)
        (tr2, rir_out2) = self.TR2(img2, rir2, mic_info, mask_pos)
        (tr3, rir_out3) = self.TR3(img3, rir3, mic_info, mask_pos)
        (tr4, rir_out4) = self.TR4(img4, rir4, mic_info, mask_pos)
        (tr5, rir_out5) = self.TR5(img5, rir5, mic_info, mask_pos)

        # print("tr1 before rescale: ", tr1.max().item(), tr1.min().item())
        # Vision rescale
        # tr1 = self._rescale_2d(tr1, img1)
        # tr2 = self._rescale_2d(tr2, img2)
        # tr3 = self._rescale_2d(tr3, img3)
        # tr4 = self._rescale_2d(tr4, img4)
        # tr5 = self._rescale_2d(tr5, img5)
        # print("tr1: ", tr1.max().item(), tr1.min().item())
        # RIR rescale
        rir_out1 = self._rescale_1d(rir_out1, rir1)
        rir_out2 = self._rescale_1d(rir_out2, rir2)
        rir_out3 = self._rescale_1d(rir_out3, rir3)
        rir_out4 = self._rescale_1d(rir_out4, rir4)
        rir_out5 = self._rescale_1d(rir_out5, rir5)

        rir_inputs  = [rir1, rir2,  rir3,  rir4,  rir5]
        rir_outputs = [rir_out1, rir_out2, rir_out3, rir_out4, rir_out5]
        rir_losses = [self.mse(x, y) for x, y in zip(rir_inputs, rir_outputs)]
        rir_loss   = sum(rir_losses) / len(rir_losses)

        out4 = self.up1(tr5, tr4) 
        out3 = self.up2(out4, tr3)  
        out2 = self.up3(out3, tr2)  
        out1 = self.up4(out2, img1)  
        
        # print("tr5 max/min:", tr5.max().item(), tr5.min().item())
        # print("tr4 max/min:", tr4.max().item(), tr4.min().item())
        # print("out4 max/min:", out4.max().item(), out4.min().item())

        # print("tr3 max/min:", tr3.max().item(), tr3.min().item())
        # print("out3 max/min:", out3.max().item(), out3.min().item())

        # print("tr2 max/min:", tr2.max().item(), tr2.min().item())
        # print("out2 max/min:", out2.max().item(), out2.min().item())

        # print("tr1 max/min:", tr1.max().item(), tr1.min().item())
        # print("out1 max/min:", img1.max().item(), img1.min().item())

        
        # print("out1: ", out1.max().item(), out1.min().item())
         
        img_rec = self.scale * self.outc(out1) + self.bias   # (B, 1, 256, 512)'
        # print("img_rec_before_clampe: ", img_rec.max().item(), img_rec.min().item())
        img_rec = torch.clamp_min(img_rec, -1029.0/410.0)
        
        # print("img_rec: ", img_rec.max().item(), img_rec.min().item())
        return img_rec, rir_loss

