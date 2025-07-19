import os

def get_folder_names(directory):
    # 지정된 디렉토리 내의 폴더 이름을 집합으로 반환합니다.
    return {name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))}

if __name__ == "__main__":
    dir1 = "/home/byulharang/MP3D_ACN_N3D/Train"   # 첫 번째 디렉토리 경로 입력
    dir2 = "/home/byulharang/MP3D_ACN_N3D/Depth/Train"  # 두 번째 디렉토리 경로 입력

    folders1 = get_folder_names(dir1)
    folders2 = get_folder_names(dir2)

    unique_to_dir1 = folders1 - folders2
    unique_to_dir2 = folders2 - folders1

    print("첫 번째 디렉토리에만 있는 폴더:")
    for folder in unique_to_dir1:
        print(folder)

    print("\n두 번째 디렉토리에만 있는 폴더:")
    for folder in unique_to_dir2:
        print(folder)
