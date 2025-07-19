import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import math


class compute_psnr(nn.Module):
    def __init__(self, max_val=1.0):
        super(compute_psnr, self).__init__()
        self.max_val = max_val
        self.mse = nn.MSELoss()

    def normalize(self, img1, img2):
        B, C, H, W = img1.shape
        flat1 = img1.view(B, -1)
        flat2 = img2.view(B, -1)
        min1 = flat1.min(1, True)[0].view(B, 1, 1, 1)
        max1 = flat1.max(1, True)[0].view(B, 1, 1, 1)
        min2 = flat2.min(1, True)[0].view(B, 1, 1, 1)
        max2 = flat2.max(1, True)[0].view(B, 1, 1, 1)
        img1 = (img1 - min1) / (max1 - min1 + 1e-8)
        img2 = (img2 - min2) / (max2 - min2 + 1e-8)
        return img1, img2

    def forward(self, img1, img2, mask_pos, mask_width):
        B, C, H, W = img1.shape
        mask_pos = torch.as_tensor(mask_pos, device=img1.device).long().view(-1)
        w = int(mask_width)
        m1 = torch.zeros((B, C, H, w), device=img1.device, dtype=img1.dtype)
        m2 = torch.zeros_like(m1)
        for b in range(B):
            s = mask_pos[b].item()
            m1[b] = img1[b, ..., s : s + w]
            m2[b] = img2[b, ..., s : s + w]
        m1, m2 = self.normalize(m1, m2)
        mse = self.mse(m1, m2)
        if mse == 0:
            return torch.tensor(float('inf'), device=img1.device)
        return 10 * torch.log10((self.max_val ** 2) / mse)


class compute_lpips(nn.Module):
    def __init__(self, lpips_net='vgg', max_val=1.0):
        super(compute_lpips, self).__init__()
        self.max_val = max_val
        self.lpips_model = lpips.LPIPS(net=lpips_net)
        self.lpips_model.eval()

    def normalize(self, img1, img2):
        B, C, H, W = img1.shape
        flat1 = img1.view(B, -1)
        flat2 = img2.view(B, -1)
        min1 = flat1.min(1, True)[0].view(B, 1, 1, 1)
        max1 = flat1.max(1, True)[0].view(B, 1, 1, 1)
        min2 = flat2.min(1, True)[0].view(B, 1, 1, 1)
        max2 = flat2.max(1, True)[0].view(B, 1, 1, 1)
        img1 = (img1 - min1) / (max1 - min1 + 1e-8)
        img2 = (img2 - min2) / (max2 - min2 + 1e-8)
        return img1, img2

    def forward(self, img1, img2, mask_pos, mask_width):
        B, C, H, W = img1.shape
        if C == 1:
            img1 = img1.repeat(1, 3, 1, 1)
            img2 = img2.repeat(1, 3, 1, 1)
        mask_pos = torch.as_tensor(mask_pos, device=img1.device).long().view(-1)
        w = int(mask_width)
        m1 = torch.zeros((B, 3, H, w), device=img1.device, dtype=img1.dtype)
        m2 = torch.zeros_like(m1)
        for b in range(B):
            s = mask_pos[b].item()
            m1[b] = img1[b, ..., s : s + w]
            m2[b] = img2[b, ..., s : s + w]
        m1, m2 = self.normalize(m1, m2)
        with torch.no_grad():
            score = self.lpips_model(m1, m2)
        return score


class SSIM(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, sigma)

    def _gaussian(self, window_size, sigma):
        gauss = torch.tensor([
            math.exp(-((x - window_size//2)**2)/(2*sigma**2))
            for x in range(window_size)
        ], dtype=torch.float32)
        return gauss / gauss.sum()

    def _create_window(self, window_size, sigma):
        _1D = self._gaussian(window_size, sigma).unsqueeze(1)
        _2D = _1D @ _1D.t()
        win = _2D.unsqueeze(0).unsqueeze(0)
        return win

    def normalize(self, img1, img2):
        B, C, H, W = img1.shape
        flat1 = img1.view(B, -1)
        flat2 = img2.view(B, -1)
        min1 = flat1.min(1, True)[0].view(B, 1, 1, 1)
        max1 = flat1.max(1, True)[0].view(B, 1, 1, 1)
        min2 = flat2.min(1, True)[0].view(B, 1, 1, 1)
        max2 = flat2.max(1, True)[0].view(B, 1, 1, 1)
        img1 = (img1 - min1) / (max1 - min1 + 1e-8)
        img2 = (img2 - min2) / (max2 - min2 + 1e-8)
        return img1, img2

    def _ssim(self, img1, img2, window):
        C = img1.size(1)
        mu1 = F.conv2d(img1, window.expand(C,1,*window.shape[-2:]), padding=self.window_size//2, groups=C)
        mu2 = F.conv2d(img2, window.expand(C,1,*window.shape[-2:]), padding=self.window_size//2, groups=C)
        mu1_sq, mu2_sq = mu1**2, mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1 = F.conv2d(img1*img1, window.expand(C,1,*window.shape[-2:]), padding=self.window_size//2, groups=C) - mu1_sq
        sigma2 = F.conv2d(img2*img2, window.expand(C,1,*window.shape[-2:]), padding=self.window_size//2, groups=C) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window.expand(C,1,*window.shape[-2:]), padding=self.window_size//2, groups=C) - mu1_mu2
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1+sigma2+C2))
        return ssim_map.mean() if self.size_average else ssim_map

    def forward(self, img1, img2, mask_pos, mask_width):
        B, C, H, W = img1.shape
        window = self.window.to(img1.device, dtype=img1.dtype)
        mask_pos = torch.as_tensor(mask_pos, device=img1.device).long().view(-1)
        w = int(mask_width)
        m1 = torch.zeros((B, C, H, w), device=img1.device, dtype=img1.dtype)
        m2 = torch.zeros_like(m1)
        for b in range(B):
            s = mask_pos[b].item()
            m1[b] = img1[b, ..., s : s + w]
            m2[b] = img2[b, ..., s : s + w]
            m1, m2 = self.normalize(m1, m2)
        return self._ssim(m1, m2, window)


class RelativeError(nn.Module):
    def __init__(self, eps=1e-8):
        super(RelativeError, self).__init__()
        self.eps = eps

    def forward(self, GT, gen, mask_pos, mask_width):
        B, C, H, W = GT.shape
        mask_pos = torch.as_tensor(mask_pos, device=GT.device).long().view(-1)
        w = int(mask_width)
        g1 = torch.zeros((B, C, H, w), device=GT.device, dtype=GT.dtype)
        g2 = torch.zeros_like(g1)
        for b in range(B):
            s = mask_pos[b].item()
            g1[b] = GT[b, ..., s : s + w]
            g2[b] = gen[b, ..., s : s + w]
    
        rel = torch.mean(torch.abs(g1 - g2) / (torch.abs(g1) + self.eps))
        return rel  
