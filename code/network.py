import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.cm as cm
from net.Hybrid_unetr import Hybrid_unetr
from loss import (
    masked_region_MSE_Loss,
    MSE_Loss,
    SSIMLoss,
    DepthGANLoss,
    Sobel_Loss,
)
from metric import compute_psnr, compute_lpips, SSIM, RelativeError


class DepthExtract(nn.Module):
    def __init__(self):
        super(DepthExtract, self).__init__()
        self.midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        for param in self.midas_model.parameters():
            param.requires_grad = False
        self.midas_model.eval()

    def forward(self, masked_img):
        masekd_depth = self.midas_model(masked_img)
        masekd_depth = F.interpolate(
            masekd_depth.unsqueeze(1), size=(256, 512), mode="bilinear", align_corners=False
        )
        return masekd_depth


class UnetModel(nn.Module):
    def __init__(self):
        super(UnetModel, self).__init__()
        self.Depth = DepthExtract()
        self.model = Hybrid_unetr()
        self.ave_depth = 1029
        self.std_depth = 410

        # losses
        self.context_loss = MSE_Loss()
        self.recon_loss = masked_region_MSE_Loss()
        self.structural_loss = SSIMLoss()
        self.DepthGAN_loss = DepthGANLoss()
        self.sobel_loss = Sobel_Loss()

        # metrics
        self.metric_psnr = compute_psnr()
        self.metric_lpips = compute_lpips()
        self.metric_ssim = SSIM()
        self.metric_rel = RelativeError()

        # scales
        
        self.loss_scale = 1 * 10e-1

        self.context_loss_scale = 2 * 10e1
        self.depth_recon_loss_scale = 1.0 * 10e2
        self.structural_loss_scale =  0 *1 * 10e1    
        self.DepthGAN_loss_scale = 0 * 1 * 10e4
        self.sobel_loss_scale = 0.0 * 10e-8       
        self.rir_recon_loss_scale = 0 * 1.0 * 10e-6 

        self.monitoring = True

    def save_depth_map(self, depth_tensor, save_path, h, w, pos_out, region_width=128, cmap='jet_r'):
        depth_array = depth_tensor.squeeze().detach().cpu().numpy().reshape((h, w))
        depth_normalized = (
            255 * (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array))
        ).astype(np.uint8)

        if cmap != '':  
            colormap = cm.get_cmap(cmap)
            depth_color = colormap(depth_normalized / 255.0)  # shape: (h, w, 4)
            depth_color = (depth_color[:, :, :3] * 255).astype(np.uint8)
            depth_image = Image.fromarray(depth_color)
            border_color = (0, 0, 0)  
        else:  # gray scale
            depth_image = Image.fromarray(depth_normalized)
            depth_image = depth_image.convert("RGB")
            border_color = (255, 0, 0)  

        draw = ImageDraw.Draw(depth_image)
        draw.rectangle(
            [pos_out, 0, pos_out + region_width, h],
            outline=border_color,
            width=3
        )
        depth_image.save(save_path)

    def find_masked_start_pos(self, masked_img, mask_width):
        B, W = masked_img.shape[0], masked_img.shape[3]
        positions = []
        for i in range(B):
            pos = -1
            for x in range(W - mask_width + 1):
                if torch.all(masked_img[i, :, :, x] == 0):
                    pos = x
                    break
            positions.append(pos)
        return torch.tensor(positions, device=masked_img.device)
 
    def forward(self, masked_img, GT_img, rir, mic_info, exp_order=None, metric=False):
        # extract mask positions
        mask_width = 256
        mask_width_depth = mask_width // 2
        mask_pos = self.find_masked_start_pos(masked_img, mask_width)  # (B,)
        mask_pos_depth = mask_pos // 2
        
        # depth preprocessing
        GT_depth = self.Depth(GT_img)
        normed_GT_depth = (GT_depth - self.ave_depth) / self.std_depth
        masked_depth = self.Depth(masked_img)
        normed_masked_depth = (masked_depth - self.ave_depth) / self.std_depth
        
        # Masked Depth Area -> Zero padding
        B, C, H, W = masked_depth.shape
        device = masked_depth.device
        
        col_idx = torch.arange(W, device=device).view(1,1,1,W)            # [1,1,1,W]
        start   = mask_pos_depth.view(B,1,1,1)                            # [B,1,1,1]
        end     = start + mask_width_depth                                # [B,1,1,1]
        mask_map = ((col_idx < start) | (col_idx >= end)).float()         # [B,1,1,W]
        mask_map = mask_map.expand(B,1,H,W) 
        normed_masked_depth = normed_masked_depth * mask_map              # [B,1,H,W]
        

        # model forward
        normed_gen_depth, rir_recon = self.model(normed_masked_depth, rir, mic_info, mask_pos) 
        gen_depth = self.std_depth * normed_gen_depth + self.ave_depth
        
        # print("normed_gen: ", normed_gen_depth.max().item(), normed_gen_depth.min().item())
        # print("normed_GT: ", normed_GT_depth.max().item(), normed_GT_depth.min().item())
        
        # overlay: masked_depth에 gen_region 덮어쓰기
        full = masked_depth.clone()
        starts = mask_pos_depth.long()
        for b in range(B):
            s = starts[b].item()
            assert 0 <= s <= full.size(-1) - mask_width_depth, "gen_region 범위를 벗어났습니다"
            full[b, :, :, s:s + mask_width_depth] = gen_depth[b, :, :, s:s + mask_width_depth]
        edit_gen_depth = full   # gen area + given masked depth
    


        if metric:
            # compute each metric and average across batch
            psnr = self.metric_psnr(GT_depth, gen_depth, mask_pos_depth, mask_width_depth)
            lpips = self.metric_lpips(GT_depth, gen_depth, mask_pos_depth, mask_width_depth)
            ssim = self.metric_ssim(GT_depth, gen_depth, mask_pos_depth, mask_width_depth)
            rel = self.metric_rel(GT_depth, gen_depth, mask_pos_depth, mask_width_depth)
            # metrics may return batch tensors; reduce to scalar
            return psnr, lpips, ssim, rel

        # training loss
        context_loss = self.context_loss_scale * self.context_loss(normed_GT_depth, normed_gen_depth)
        recon_loss = self.depth_recon_loss_scale * self.recon_loss(normed_GT_depth, normed_gen_depth, mask_pos_depth, mask_width_depth)
        structural_loss = self.structural_loss_scale * self.structural_loss(normed_GT_depth, normed_gen_depth, mask_pos_depth, mask_width_depth)
        DepthGAN_loss = self.DepthGAN_loss_scale * self.DepthGAN_loss(normed_GT_depth, normed_gen_depth, mask_pos_depth, mask_width_depth)
        rir_loss = rir_recon * self.rir_recon_loss_scale    
        sobel_loss = self.sobel_loss_scale * self.sobel_loss(normed_GT_depth, normed_gen_depth, mask_pos_depth, mask_width_depth)
        total_loss = self.loss_scale * (context_loss + recon_loss)
        
        print(f"conetext loss: {context_loss.item():.4f}," 
              f"recon loss: {recon_loss.item():.4f}," 
              f"SSIM loss: {structural_loss.item():.4f},"
              f"GAN loss: {DepthGAN_loss.item():.4f},"
              f"RIR loss: {rir_loss.item():.4f}," 
              f"sobel loss: {sobel_loss.item():.4f},"
              f"total loss: {total_loss.item():.4f}")


        if self.monitoring and exp_order is not None:
            save_dir = "/workspace/monitoring"
            os.makedirs(save_dir, exist_ok=True)
            self.save_depth_map(
                edit_gen_depth[0], f"{save_dir}/EDIT_GEN_{exp_order}.png", 256, 512, mask_pos_depth[0], mask_width_depth
            )
            self.save_depth_map(
                normed_gen_depth[0], f"{save_dir}/GEN_{exp_order}.png", 256, 512, mask_pos_depth[0], mask_width_depth
            )
            self.save_depth_map(
                GT_depth[0],  f"{save_dir}/GT_{exp_order}.png",  256, 512, mask_pos_depth[0], mask_width_depth
            )

        return total_loss
