import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from PIL import Image
from setproctitle import setproctitle
setproctitle("Depth_Extract")


## setting ##

input_dir = '/home/byulharang/MP3D_ACN_N3D/Train9'
matrix_dir = '/home/byulharang/MP3D_ACN_N3D/Depth/Train9' # matrix

os.makedirs(matrix_dir, exist_ok=True)

    # GPU Select   
os.environ['CUDA_VISIBLE_DEVICES'] = '9'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
model_type = "DPT_Hybrid"  # DPT_Large, DPT_Hybrid

    # MiDaS Load to CUDA
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device) 

# Transform to convert PIL image to tensor
to_tensor = transforms.ToTensor()

image_formats = ('.png')

## Processing ##

for room_name in os.listdir(input_dir):
    room_dir = os.path.join(input_dir, room_name)
    for filename in os.listdir(room_dir):
        if filename.lower().endswith(image_formats): 
             
            # get image
            image_path = os.path.join(input_dir, room_name, filename)   
            image = Image.open(image_path)
            if image.mode == 'RGBA':
                image = image.convert('RGB')  # Convert RGBA to RGB if necessary
                
            # Convert image to tensor and normalize
            image = to_tensor(image).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension (1, 3, H, W)

            # Depth Estimation
            with torch.no_grad():
                depth = midas(image)

            # Post Processing
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=image.shape[-2:],  # Get height and width from the image tensor
                mode="bicubic",
                align_corners=False,
            )   # [1, 1, 512, 1024]
            
            AvePool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2,2))
            depth = AvePool(depth) 
            # depth = maxpool(depth) 
            # depth = maxpool(depth) 
            # depth = maxpool(depth)
            # depth = maxpool(depth) # [1 x 1 x 16 x 32]
            depth = depth.squeeze(0)
            depth = depth.squeeze(0)
            depth = depth.cpu().numpy()
            
            # Save Part (Graph, Matrix)
            match = re.search(r'\d+', filename)
            if match:
                number = match.group(0)  # Extract only the number part
            else:
                number = "unknown"  
                
            # Save Matrix
            matrix_room_dir = os.path.join(matrix_dir, room_name)
            os.makedirs(matrix_room_dir, exist_ok=True)
            
            depth_mat_path = os.path.join(matrix_room_dir, f'{number}_depth.csv')
            np.savetxt(depth_mat_path, depth, delimiter=",")
            print(f"Saved depth to {depth_mat_path}.")

print("All process Done Successfully depth matrix saved.")
