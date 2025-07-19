import numpy as np
import os
import re
from CRIP.preprocessing_tools.audio2stft import n_fft, sr

## setting ##

phase_directory = '/home/byulharang/preprocessing/audio_raw_phase'     
wave_length_out_path = '/home/byulharang/CRIP/phase_enhance_tool.csv'
enhanced_phase_out_path = '/home/byulharang/preprocessing/Enhanced_phase'
os.makedirs(enhanced_phase_out_path, exist_ok=True)

format = ('.csv')
speed = 343   # m/s (297K Temperature)

## Freq bins ##

freqs = np.fft.rfftfreq(n_fft, d=1/sr)
freqs = freqs + 1e-6    # Prevent div 0Hz

## Wave Length ##

wave_length = (speed / freqs)   # [0]:  0Hz, [512] : 8kHz
wave_length = np.round(wave_length, 8)      # 소숫점 8자리 반올림
wave_length[0] = 1
np.savetxt(wave_length_out_path, wave_length, delimiter=',')

## inhance phase ##



for filename in os.listdir(phase_directory):
    if filename.lower().endswith(format):  
        phase_path = os.path.join(phase_directory, filename)
        phase = np.genfromtxt(phase_path, delimiter=',')   # Get Phase 1 by 1
        
            # give weight to phase
        enhaced_phase = []
        for i in range(len(wave_length)):
            enhaced_phase.append(phase[i, :] * wave_length[512-i])
            
        enhaced_phase = np.vstack(enhaced_phase)    # Get Enhanced Phase
            
        ##  Save Part   ##
        
        match = re.search(r'\d+', filename)
        if match:
            number = match.group(0)
        else:
            number = "unknown"  
        
            # Save
        output_path = os.path.join(enhanced_phase_out_path, f'{number}_Enhanced_phase.csv')
        np.savetxt(output_path, enhaced_phase, delimiter=',')
        
    print(f"Saved Enhanced_Phase to {output_path}.")
print("All Enhanced Phase Processing Done Successfully")

