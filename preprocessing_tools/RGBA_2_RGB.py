import os
from PIL import Image

# 원본 이미지들이 있는 디렉토리 경로
input_directory = "/home/byulharang/dataset/gTV8FGcVJC9"
# 변환된 이미지를 저장할 디렉토리 경로
output_directory = "/home/byulharang/image_edit_dataset"

# 이미지 파일 포맷 리스트 (확장자)
image_formats = ('.png')

# 디렉토리 내 모든 파일에 대해 반복 작업
for filename in os.listdir(input_directory):
    if filename.lower().endswith(image_formats):  # 이미지 파일인지 확인
        # 이미지 파일 경로
        image_path = os.path.join(input_directory, filename)
        
        # 이미지 로드
        image = Image.open(image_path)
        
        # 이미지가 4채널(RGBA)인 경우 3채널(RGB)로 변환
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            print(f"Converted {filename} from RGBA to RGB.")

        # 저장할 경로 설정
        output_path = os.path.join(output_directory, filename)
        
        # 이미지 저장
        image.save(output_path)
        print(f"Saved converted image to {output_path}.")

print("All images processed and saved.")
