import torch
import torchvision.transforms as transforms
from PIL import Image
from crip_HorizonNet import ImageEmbeddingNet
import numpy as np

# 모델 초기화
model = ImageEmbeddingNet()
model.eval()  # 평가 모드로 전환 (필요한 경우)

# 이미지를 읽고 변환 적용
image_path = 'dataset/mp3d/5q7pvUzZiYa/0-rgb.png'
image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((512, 1024)),  # 필요한 크기로 조정
    transforms.ToTensor(),  # 이미지를 텐서로 변환
])

image_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가 (1, C, H, W)

# 모델을 사용하여 결과를 얻기
with torch.no_grad():  # 평가 모드에서 gradient 계산 비활성화
    top = model(image_tensor)
    print(top)

# 모델 출력 후처리
# 모델에서 나오는 top이 텐서인 경우, 이를 numpy 배열로 변환 및 정수형으로 변환
top = top.squeeze().cpu().numpy()  # 텐서 -> numpy 배열로 변환
top = (top + 0.5) * (512 // 2)
top = top.astype(np.int32)
print(top)

image_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

# 이미지에 빨간 선을 추가
for x in range(1024):
    y = top[x]  # top은 2차원 배열이므로 첫 번째 차원에 접근
    if 0 <= y < image_np.shape[0]:  # y값이 이미지의 높이 범위 내에 있는지 확인
        image_np[y, x, 0] = 1.0  # 빨간색 채널을 최대값으로 설정 (0~1 범위)
        image_np[y, x, 1] = 0.0
        image_np[y, x, 2] = 0.0

# 변환된 이미지를 저장
modified_image = Image.fromarray(np.uint8(image_np * 255))
modified_image.save('modified_image_with_red_lines.png')
