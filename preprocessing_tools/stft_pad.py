import os
import re
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import CRIP.edit_audio_matrix as edit_audio_matrix


Real_dir = '/home/byulharang/Dataset_Preproc/Real_data'  # real part
Imaginary_dir = '/home/byulharang/Dataset_Preproc/Imaginary_data'    # imaginary part


## Processing ##
if __name__ == "__main__":
    for room_name in os.listdir((Real_dir)):

        Room_Real_dir = os.path.join(Real_dir, room_name)
        Room_Imag_dir = os.path.join(Imaginary_dir, room_name)

        for real_file, imag_file in zip(sorted(os.listdir(Room_Real_dir)), sorted(os.listdir(Room_Imag_dir))):
            real_path = os.path.join(Room_Real_dir, real_file)
            imag_path = os.path.join(Room_Imag_dir, imag_file)
                
            real_data = np.genfromtxt(real_path, delimiter=',')
            imag_data = np.genfromtxt(imag_path, delimiter=',')
                
            real_edit = edit_audio_matrix.pad_mat_stft(real_data)
            imag_edit = edit_audio_matrix.pad_mat_stft(imag_data)
        
            np.savetxt(real_path, real_edit, delimiter=",")
            np.savetxt(imag_path, imag_edit, delimiter=",")
            print(real_edit.shape, imag_edit.shape)
            print(f"Saved {real_path}.")
            print(f"Saved {imag_path}.")
            print('-'*20)

    print("All audio files processed and Matrix saved.")
