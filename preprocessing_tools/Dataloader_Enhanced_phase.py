import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet50_Weights
from torchvision import models
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm
from PIL import Image

import ckpt_horizon
import edit_audio_matrix
import torchaudio
from HorizonNet.model import HorizonNet
from HorizonNet.misc import utils
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)


class PairedDataset(Dataset):
    def __init__(self, Image_dir, Audio_dir, transform=None, n_fft=320, hop_length=160):
        self.file_list = []
        self.root_dirs = [Image_dir, Audio_dir]
        self.root_dir = Image_dir
        self.transform = transform if transform else transforms.ToTensor()  # 기본 변환 추가
        self.n_fft = n_fft
        self.hop_length = hop_length

        # extract room directories
        self.room_dirs = os.listdir(self.root_dir)
        for room_name in self.room_dirs:
            image_files = []
            audio_files = []
            folder_path = os.path.join(self.root_dir, room_name)
            # Get all .png files in the directory
            image_file_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])
            image_files.extend(image_file_paths)
            # Get all .wav files in the directory
            audio_file_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')])
            audio_files.extend(audio_file_paths)                                 
            # Ensure that the number of image and audio files match
            if len(image_files) == len(audio_files):
                for img_file, audio_file in zip(image_files, audio_files):
                    self.file_list.append((img_file, audio_file))   # 1 Idx = 1 Pair
            else:
                raise ValueError(f"Number of image and audio files do not match in {room_name}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, audio_path = self.file_list[idx]

        # Imag
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)  # 이미지 변환 (PIL -> Tensor)
        image = image.unsqueeze(0)  # (3, 512, 1024) -> (1, 3, 512, 1024)로 변경
        
        # Audio 1D
        B_format, sr = torchaudio.load(audio_path)
        if B_format.shape[0] == 4:
            waveform = B_format[0, :].unsqueeze(0)  # W Channel for Mono

        # STFT 
        window = torch.hann_window(self.n_fft)
        stft_result = torch.stft(waveform, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=window,
            win_length=self.n_fft, 
            return_complex=True
        )
        
        real_part = stft_result.real  
        imag_part = stft_result.imag  
        
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.atan2(imag_part, real_part)
        
            # phase Enhance
        freqs = np.fft.rfftfreq(self.n_fft, d=1/sr)
        freqs = freqs + 1e-6    # Prevent div 0Hz


            # wavelength table
        wave_length = (343 / freqs)   # [0]:  0Hz, [512] : 8kHz
        wave_length = np.round(wave_length, 6)      # 소숫점 6자리 반올림
        wave_length[0] = 1
        wave_length_tensor = torch.tensor(wave_length)
            
        enhanced_phase = []
        for i in range(len(wave_length)):
            enhanced_phase.append(phase[0, i, :] * wave_length_tensor[160 - i])
            
        enhanced_phase_tensor = torch.stack(enhanced_phase)
        
        Real = edit_audio_matrix.pad_mat_stft(real_part.squeeze(0)) # 161 x 300
        Imag = edit_audio_matrix.pad_mat_stft(imag_part.squeeze(0)) # 161 x 300
        
        #Mag = edit_audio_matrix.pad_mat_stft(magnitude.squeeze(0)) # 161 x 300
        #Phase = edit_audio_matrix.pad_mat_stft(phase.squeeze(0)) # 161 x 300
        #Enhanced_phase = edit_audio_matrix.pad_mat_stft(enhanced_phase_tensor.squeeze(0)) # 161 x 300
        
        return image, Real, Imag
        #return image, Mag, Enhanced_phase # Mag-> dummy
