import cv2
import numpy as np
import matplotlib.pyplot as plt

## setting ##

pad_size = 1024
resize_size = (224, 224)

# padding & resizing (512 x 1024 -> 1024 x 1024 -> 224 x 224)  (questionable)
# 추가하게 되면 데이터 vertical 데이터 손실이 커짐. 공간적 정보를 보존하는 측면에서 왜곡된 것이 성능을 저하시킬까? 과연
# zero-padding only right side & cubic interpolation

def edit_depth_mat(matrix):

    pad_matrix = pad_mat_depth(matrix)
    resize_matrix = resize_mat_depth(pad_matrix)

    edit_matrix = resize_matrix
    return edit_matrix


def pad_mat_depth(matrix):

    # zero-padding to be sqaure (Farter -> zero in MiDaS)
    pad_vertical = (pad_size - matrix.shape[0]) // 2  # Vertical Pad, No Need for Horizontal

    # padding
    padded_matrix = np.pad(matrix, ((pad_vertical, pad_vertical), (0, 0)), mode='constant', constant_values=0)

    return padded_matrix

def resize_mat_depth(matrix):
    resized_mat = cv2.resize(matrix, resize_size, interpolation=cv2.INTER_AREA)

    return resized_mat



target = np.loadtxt('./depth.csv', delimiter=',')
editted_target = edit_depth_mat(target)
np.savetxt('/home/byulharang/pad_resize_depth.csv', editted_target, delimiter=',')
print(f"After all, {editted_target.shape}")

# Visualization
plt.imshow(target, cmap='inferno')
plt.colorbar()
plt.show()
plt.savefig('/home/byulharang/depth.png')
