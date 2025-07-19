import os
import re
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import CRIP.edit_audio_matrix as edit_audio_matrix

## setting ##

    # directory path Select
Input_dir = '/home/byulharang/dataset/mp3d'  # Audio 1D
Spec_dir = '/home/byulharang/Dataset_Preproc/Spectrogram'  # spectrogram
Mag_dir = '/home/byulharang/Dataset_Preproc/Mag_data'    # magnitude
Phase_dir = '/home/byulharang/Dataset_Preproc/Phase_data'    # phase
Real_dir = '/home/byulharang/Dataset_Preproc/Real_data'  # real part
Imaginary_dir = '/home/byulharang/Dataset_Preproc/Imaginary_data'    # imaginary part

    # make directory
#os.makedirs(Spec_dir, exist_ok=True)
#os.makedirs(Mag_dir, exist_ok=True)
#os.makedirs(Phase_dir, exist_ok=True)
os.makedirs(Real_dir, exist_ok=True)
os.makedirs(Imaginary_dir, exist_ok=True)

    # AudioLDM, sr = 16K, below corr.
audio_formats = ('.wav')
n_fft = 1024    
hop_length = 160
sr = 16000

    # GPU setting   BASH:  export CUDA_VISIBLE_DEVICES=8
os.environ['CUDA_VISIBLE_DEVICES'] = '8'
device_ids = [0]   # 실제 번호와 상관없이 0부터 시작 or check할 것
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Processing ##
if __name__ == "__main__":
    for room_name in os.listdir((Input_dir)):
        room_dir = os.path.join(Input_dir, room_name)
        
        Room_Real_dir = os.path.join(Real_dir, room_name)
        Room_Imag_dir = os.path.join(Imaginary_dir, room_name)
        
        os.makedirs(Room_Real_dir, exist_ok=True)
        os.makedirs(Room_Imag_dir, exist_ok=True)
        
        for filename in os.listdir(room_dir):
            if filename.lower().endswith(audio_formats):  
           
                # get audio
                audio_path = os.path.join(room_dir, filename)   
                B_format, sr = torchaudio.load(audio_path)
                if B_format.shape[0] == 4:
                    waveform = B_format[0, :].unsqueeze(0)  # W Channel for Mono
                waveform = waveform.to(device)  # Move to GPU
                
                # STFT
                stft = torch.stft(
                    waveform,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    return_complex=True    
                )
                real = stft.real.cpu().numpy().squeeze(0)
                imaginary = stft.imag.cpu().numpy().squeeze(0)
                real = edit_audio_matrix.pad_mat_stft(real)
                imaginary = edit_audio_matrix.pad_mat_stft(imaginary)
                print(real.shape)
                
                    # Save Part
                match = re.search(r'\d+', filename)
                if match:
                    number = match.group(0)     # extract only Number part
                else:
                    number = "unknown"  
                
                    # Save
                #output_path = os.path.join(Spec_dir, f'{number}_spec.png')
                '''
                # Design Graph
                plt.figure(figsize=(10, 6))
                librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'Spectrogram of {number}')
                plt.savefig(output_path)
                plt.close()
                '''
                
                '''     mag, phase
                magnitude_path = os.path.join(save_mag_directory, f'{number}_stft_magnitude_matrix.csv')
                phase_path = os.path.join(save_phase_directory, f'{number}_stft_phase_matrix.csv')
                
                np.savetxt(magnitude_path, edit_magnitude, delimiter=",")
                np.savetxt(phase_path, edit_phase, delimiter=",")
                
                print(f"Saved magnitude to {magnitude_path}.")
                print(f"Saved phase to {phase_path}.")
                ''' 
                
                '''     real, imag      '''

                real_path = os.path.join(Room_Real_dir, f'{number}_real.csv')
                imag_path = os.path.join(Room_Imag_dir, f'{number}_imag.csv')        

                np.savetxt(real_path, real, delimiter=",")
                np.savetxt(imag_path, imaginary, delimiter=",")

                print(f"Saved {real_path}.")
                print(f"Saved {imag_path}.")
                print('-'*20)

    print("All audio files processed and Matrix saved.")
