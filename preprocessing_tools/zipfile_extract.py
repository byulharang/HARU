import zipfile
import os

def extract_zip(zip_path, extract_to):
    # 압축 해제할 디렉토리가 없으면 생성합니다.
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    # zip 파일 열고 압축 해제
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"압축 해제 완료: {extract_to}")

if __name__ == "__main__":
    zip_path = "/home/byulharang/MP3D_ACN_N3D_L.zip"  # 압축 파일의 경로를 입력하세요.
    extract_to = "/home/byulharang/dataset/Train"  # 압축 해제할 디렉토리 경로를 입력하세요.
    
    extract_zip(zip_path, extract_to)
