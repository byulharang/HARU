import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random


class PairedDataset(Dataset):
    def __init__(self, dataset, transform=None, n_fft=256, hop=16):
        self.file_list = []
        self.root_dir = dataset
        self.img_transform = transforms.Compose([
                transforms.Resize((512, 1024)),
                transforms.ToTensor(),
        ])
        self.n_fft = n_fft
        self.hop = hop
        self.dB_convert = torchaudio.transforms.AmplitudeToDB(
            stype="magnitude", top_db=80
        )

        # extract room directories
        self.room_dirs = os.listdir(self.root_dir)
        for room_name in self.room_dirs:
            image_files = []
            audio_files = []
            mic_info_list = []

            room_path = os.path.join(self.root_dir, room_name)

            image_file_per_room = os.listdir(room_path)
            image_file_sorted = sorted(
                [f for f in image_file_per_room if f.endswith(".png")],
                key=lambda x: int(x.split("-")[0]),
            )
            for file in image_file_sorted:
                image_files.append(os.path.join(room_path, file))

            audio_file_per_room = os.listdir(room_path)
            audio_file_sorted = sorted(
                [f for f in audio_file_per_room if f.endswith(".wav")],
                key=lambda x: int(x.split("-")[0]),
            )
            for file in audio_file_sorted:
                audio_files.append(os.path.join(room_path, file))

            json_files = [f for f in audio_file_per_room if f.endswith(".json")]
            if json_files:
                mic_json_path = os.path.join(room_path, json_files[0])
                with open(mic_json_path, "r") as f:
                    mic_data = json.load(f)

                sorted_keys = sorted(mic_data.keys(), key=lambda x: int(x))
                for key in sorted_keys:
                    mic_info_list.append(mic_data[key])
            else:
                mic_info_list = [None] * len(audio_files)
            if (
                len(image_files) == len(audio_files)
                and len(image_files) == len(mic_info_list)
            ):
                for idx, (img_file, audio_file) in enumerate(
                    zip(image_files, audio_files)
                ):
                    mic_info = mic_info_list[idx]
                    self.file_list.append((img_file, audio_file, mic_info))
            else:
                raise ValueError(
                    f"Number of image, audio, depth, and mic info files do not match in {room_name}"
                )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, audio_path, mic_info = self.file_list[idx]

        # Image load
        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)  # Tensor

        # Masked Image
        masked_image = image.clone()
        mask_start = random.randint(0, 1024 - 256)
        masked_image[:, :, mask_start : mask_start + 256] = 0

        # Audio load
        B_format, sr = torchaudio.load(audio_path)
        _, current_length = B_format.shape
        padding = 8192 - current_length
        if padding > 0:
            B_format = F.pad(B_format, (0, padding))
        else:
            B_format = B_format

        W = B_format[0, :8192]
        Y = B_format[1, :8192]
        Z = B_format[2, :8192]
        X = B_format[3, :8192]
        rirs_1D = [W, Y, Z, X]
        rirs_2D = []
        window = torch.hann_window(self.n_fft)
        for rir in rirs_1D:
            stft_result = torch.stft(
                rir,
                n_fft=self.n_fft,
                hop_length=self.hop,
                window=window,
                win_length=self.n_fft,
                return_complex=True,
            )
            rirs_2D.append(stft_result)

        reals = [stft.real for stft in rirs_2D]
        imags = [stft.imag for stft in rirs_2D]
        mags = [torch.sqrt(real**2 + imag**2) for real, imag in zip(reals, imags)]
        mag_dBs = [self.dB_convert(mag) for mag in mags]
        mag_dBs = [mag - mag.min() for mag in mag_dBs]
        phases = [torch.atan2(imag, real) for real, imag in zip(reals, imags)]

        mag_dBs_tensor = torch.stack(mag_dBs, dim=0)
        phases_tensor = torch.stack(phases, dim=0)
        reals_tensor = torch.stack(reals, dim=0)
        imags_tensor = torch.stack(imags, dim=0)
        rirs_mag_phase = torch.cat([mag_dBs_tensor, phases_tensor], dim=0)
        rirs_real_imag = torch.cat([reals_tensor, imags_tensor], dim=0)
        rirs_1D_tensor = torch.stack(rirs_1D, dim=0)

        mic_info = torch.tensor(mic_info, dtype=torch.float32)

        return masked_image, image, rirs_1D_tensor, mic_info
