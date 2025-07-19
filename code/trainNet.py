import torch
import torch.nn as nn
from tqdm import tqdm
import time
from torch.amp import autocast, GradScaler



class Trainer:
    def __init__(
        self, model, train_loader, val_loader, test_loader, optimizer, device, max_norm=1.0, accumulation_steps=1
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.max_norm = max_norm
        self.accumulation_steps = accumulation_steps
        self.scaler = GradScaler()

    def train(self, epoch, total_epochs, exp_order, metric=False):
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        for batch_idx, (masked_img, img, rir, mic_info) in enumerate(
            tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}")
        ):
        
            masked_img = masked_img.to(self.device)
            img = img.to(self.device)
            rir = rir.to(self.device)
            mic_info = mic_info.to(self.device)

            with autocast(device_type=self.device.type):
                loss = self.model(masked_img, img, rir, mic_info, exp_order, metric)
                loss = loss.mean() / self.accumulation_steps
            self.scaler.scale(loss).backward()
            running_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

        if (batch_idx + 1) % self.accumulation_steps != 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()


        loss = running_loss / num_batches
        return loss

    def validation(self, epoch, total_epochs, exp_order, metric=False):
        self.model.eval()
        running_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, (masked_img, img, rir, mic_info) in enumerate(
                tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{total_epochs}")
            ):
                masked_img = masked_img.to(self.device)
                img = img.to(self.device)
                rir = rir.to(self.device)
                mic_info = mic_info.to(self.device)

                loss = self.model(
                    masked_img, img, rir, mic_info, exp_order, metric
                )
                loss = loss.mean()
                running_loss += loss.item()
                num_batches += 1
            loss = running_loss / num_batches
            return loss
        
    
    def test(self, epoch, total_epochs, exp_order, metric=False, monitor=False):
        self.model.eval()
        running_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, (masked_img, img, rir, mic_info) in enumerate(
                tqdm(self.test_loader, desc=f"Epoch {epoch + 1}/{total_epochs}")
            ):
                masked_img = masked_img.to(self.device)
                img = img.to(self.device)
                rir = rir.to(self.device)
                mic_info = mic_info.to(self.device)

                loss = self.model(
                    masked_img, img, rir, mic_info, exp_order, metric
                )
                loss = loss.mean()
                running_loss += loss.item()
                num_batches += 1
                if monitor:
                    time.sleep(3)
            loss = running_loss / num_batches
            return loss

    def metrics(self, epoch, total_epochs, exp_order, metric=True):
        self.model.eval()
        running_PSNR = 0.0
        running_LPIPS = 0.0
        running_SSIM = 0.0
        running_Rel = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, (masked_img, img, rir, mic_info) in enumerate(
                tqdm(self.test_loader, desc=f"Epoch {epoch + 1}/{total_epochs}")
            ):
                masked_img = masked_img.to(self.device)
                img = img.to(self.device)
                rir = rir.to(self.device)
                mic_info = mic_info.to(self.device)

                PSNR, LPIPS, SSIM, Rel = self.model(
                    masked_img, img, rir, mic_info, exp_order, metric
                )
                PSNR = PSNR.mean()
                LPIPS = LPIPS.mean()
                SSIM = SSIM.mean()
                Rel = Rel.mean()

                running_PSNR += PSNR.item()
                running_LPIPS += LPIPS.item()
                running_SSIM += SSIM.item()
                running_Rel += Rel.item()
                num_batches += 1

            avg_LPIPS = running_LPIPS / num_batches
            avg_PSNR = running_PSNR / num_batches
            avg_SSIM = running_SSIM / num_batches
            avg_Rel = running_Rel / num_batches
            print([avg_PSNR, avg_LPIPS, avg_SSIM, avg_Rel])
            return [avg_PSNR, avg_LPIPS, avg_SSIM, avg_Rel]


class CheckpointManager:
    def __init__(self, model, optimizer, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def load_checkpoint(self, ckpt_path):
        print(f"Loading checkpoint from: {ckpt_path}")
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device)

            # 전체 모델 파라미터 로드
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(
                    checkpoint["model_state_dict"], strict=False
                )
            else:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

            # 옵티마이저 상태 로드
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except KeyError:
                print(
                    "Optimizer state not found in checkpoint. Continuing without loading optimizer state."
                )
            except Exception as e:
                print(f"Unexpected error while loading optimizer state: {e}")

            epoch = checkpoint.get("epoch", 0)
            print(f"Checkpoint successfully loaded. Starting from epoch {epoch + 1}.")
            return epoch + 1

        except FileNotFoundError:
            print(f"Checkpoint not found at path: {ckpt_path}. Starting from scratch.")
        except KeyError as e:
            print(f"KeyError in checkpoint loading: {e}")
        except Exception as e:
            print(f"Unexpected error in checkpoint loading: {e}")
        return 0

    def save_checkpoint(self, ckpt_path, epoch):
        try:
            if isinstance(self.model, torch.nn.DataParallel):
                torch.save(
                    {
                        "model_state_dict": self.model.module.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    ckpt_path,
                )
            else:
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    ckpt_path,
                )
            print(f"Checkpoint saved at epoch {epoch}.")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
