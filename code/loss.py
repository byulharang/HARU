import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils as utils


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            np.exp(-((x - window_size // 2) ** 2) / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel, device):
        _1D = self.gaussian(window_size, sigma=1.5).unsqueeze(1).to(device)
        _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D.expand(channel, 1, window_size, window_size).contiguous()

    def ssim(self, img1, img2, window_size, size_average, val_range):
        if val_range is None:
            L = 1.0 if img1.max().item() <= 1 else 1.0  # 고정
        else:
            L = val_range
        pad = window_size // 2
        _, C, _, _ = img1.size()
        window = self.create_window(window_size, C, img1.device)
        mu1 = F.conv2d(img1, window, padding=pad, groups=C)
        mu2 = F.conv2d(img2, window, padding=pad, groups=C)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2
        # print(mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12)
        
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2) + 10e-6) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 10e-6)
        # print(((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)))
        # print(((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 10e-6))
                   
        return ssim_map.mean() if size_average else ssim_map

    def forward(self, img1, img2, mask_pos, mask_width):
        B, C, H, W = img1.shape
        mask_pos = torch.as_tensor(mask_pos, device=img1.device).long().view(-1)
        w = int(mask_width)
        masked1 = torch.zeros((B, C, H, w), device=img1.device, dtype=img1.dtype)
        masked2 = torch.zeros((B, C, H, w), device=img2.device, dtype=img2.dtype)
        for b in range(B):
            s = max(0, min(mask_pos[b].item(), W - w))
            masked1[b] = img1[b, :, :, s:s + w]
            masked2[b] = img2[b, :, :, s:s + w]
        ssim_val = self.ssim(masked1, masked2, self.window_size, self.size_average, self.val_range)
        return 1 - ssim_val


class DepthGANLoss(nn.Module):
    def __init__(self, use_spectral_norm=True):
        super(DepthGANLoss, self).__init__()
        def conv_block(in_c, out_c, ks, st, pd, use_norm=True):
            layers = []
            conv = nn.Conv2d(in_c, out_c, ks, st, pd)
            if use_spectral_norm:
                conv = utils.spectral_norm(conv)
            layers.append(conv)
            if use_norm:
                layers.append(nn.InstanceNorm2d(out_c, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        c1 = nn.Conv2d(1, 64, 4, 2, 1)
        if use_spectral_norm:
            c1 = utils.spectral_norm(c1)
        layers += [c1, nn.LeakyReLU(0.2, inplace=True)]
        layers += conv_block(64, 128, 4, 2, 1)
        layers += conv_block(128, 256, 4, 2, 1)
        layers += conv_block(256, 512, 4, 1, 1, use_norm=False)

        fconv = nn.Conv2d(512, 1, 4, 1, 1)
        if use_spectral_norm:
            fconv = utils.spectral_norm(fconv)
        layers += [fconv]

        self.discriminator = nn.Sequential(*layers)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, GT, pred, mask_pos, mask_width):
        B, C, H, W = GT.shape
        mask_pos = torch.as_tensor(mask_pos, device=GT.device).long().view(-1)
        w = int(mask_width)
        gt_m = torch.zeros((B, C, H, w), device=GT.device, dtype=GT.dtype)
        pr_m = torch.zeros((B, C, H, w), device=pred.device, dtype=pred.dtype)
        for b in range(B):
            s = max(0, min(mask_pos[b].item(), W - w))
            gt_m[b] = GT[b, :, :, s:s + w]
            pr_m[b] = pred[b, :, :, s:s + w]

        real = self.discriminator(gt_m)
        fake = self.discriminator(pr_m)
        real = torch.clamp(real, min=-10, max=10)
        fake = torch.clamp(fake, min=-10, max=10)
        lr = self.bce_loss(real, torch.ones_like(real))
        lf = self.bce_loss(fake, torch.zeros_like(fake))
        return lr + lf


class masked_region_MSE_Loss(nn.Module):
    def __init__(self):
        super(masked_region_MSE_Loss, self).__init__()

    def forward(self, x, y, mask_pos, mask_width):
        B, C, H, W = x.shape
        mask_pos = torch.as_tensor(mask_pos, device=x.device).long().view(-1)
        w = int(mask_width)
        mx = torch.zeros_like(x)
        my = torch.zeros_like(y)
        for b in range(B):
            s = max(0, min(mask_pos[b].item(), W - w))
            mx[b, :, :, s:s+w] = x[b, :, :, s:s+w]
            my[b, :, :, s:s+w] = y[b, :, :, s:s+w]
        return F.mse_loss(mx, my)


class MSE_Loss(nn.Module):
    def __init__(self): super(MSE_Loss, self).__init__()
    def forward(self, x, y): return F.mse_loss(x, y)


class Sobel_Loss(nn.Module):
    def __init__(self): super(Sobel_Loss, self).__init__()

    def get_sobel_filters(self, device):
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device)
        sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device)
        sx = sx.view(1, 1, 3, 3)
        sy = sy.view(1, 1, 3, 3)
        return sx, sy

    def compute_sobel(self, img):
        B, C, H, W = img.shape
        sx, sy = self.get_sobel_filters(img.device)
        sx = sx.repeat(C, 1, 1, 1)
        sy = sy.repeat(C, 1, 1, 1)
        gx = F.conv2d(img, sx, padding=1, groups=C)
        gy = F.conv2d(img, sy, padding=1, groups=C)
        return gx, gy

    def forward(self, img1, img2, mask_pos, mask_width, loss_type='L2'):
        B, C, H, W = img1.shape
        mask_pos = torch.as_tensor(mask_pos, device=img1.device).long().view(-1)
        w = int(mask_width)
        m1 = torch.zeros((B, C, H, w), device=img1.device, dtype=img1.dtype)
        m2 = torch.zeros((B, C, H, w), device=img2.device, dtype=img2.dtype)
        for b in range(B):
            s = max(0, min(mask_pos[b].item(), W - w))
            m1[b] = img1[b, :, :, s:s+w]
            m2[b] = img2[b, :, :, s:s+w]
        gx1, gy1 = self.compute_sobel(m1)
        gx2, gy2 = self.compute_sobel(m2)
        if loss_type == 'L1':
            return F.l1_loss(gx1, gx2) + F.l1_loss(gy1, gy2)
        return F.mse_loss(gx1, gx2) + F.mse_loss(gy1, gy2)
