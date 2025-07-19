import cv2
import os
import re
import numpy as np


## setting ##

    # Entire Depth Load     
input_path = '/home/byulharang/preprocessing/img_DPT_Hybrid/'           # img_DPT_Hybrid    img_DPT_Large
    # depth save path
depth_low_dir = '/home/byulharang/preprocessing/depth_separate/low'     # Depth_Low: Longest Distance
os.makedirs(depth_low_dir, exist_ok=True)
depth_med_dir = '/home/byulharang/preprocessing/depth_separate/med'     # Depth_Med
os.makedirs(depth_med_dir, exist_ok=True)
depth_high_dir = '/home/byulharang/preprocessing/depth_separate/high'   # Depth_High: Shortest Distance
os.makedirs(depth_high_dir, exist_ok=True)

depth_formats = ('.csv')

for filename in os.listdir(input_path):
        if filename.lower().endswith(depth_formats):  

                # Get Depth
            depth_path = os.path.join(input_path, filename)   
            depth = np.genfromtxt(depth_path, delimiter=',')
            
                # Make Range (Threshold)
            min_val = np.min(depth)
            max_val = np.max(depth)
            range_low = (min_val, min_val + (max_val - min_val) // 3)            
            range_med = (range_low[1] + 1, min_val + 2 * (max_val - min_val) // 3)
            range_high = (range_med[1] + 1, max_val)
            
            ## Final OUTPUT ## 
            
                # Make Image (Separate)     0: Infinity Depth   i.e. No Effect on RIR
            dpt_low = np.where((depth >= range_low[0]) & (depth <= range_low[1]), depth, 0)
            dpt_med = np.where((depth >= range_med[0]) & (depth <= range_med[1]), depth, 0)
            dpt_high = np.where((depth >= range_high[0]) & (depth <= range_high[1]), depth, 0)
            
            ## Save ##
            
            match = re.search(r'\d+', filename)
            if match:
                number = match.group(0)     # extract only Number part
            else:
                number = "unknown"  
                
                # Save Path
            low_save_path = os.path.join(depth_low_dir, f'{number}_depth_low.csv')
            med_save_path = os.path.join(depth_med_dir, f'{number}_depth_med.csv')                
            high_save_path = os.path.join(depth_high_dir, f'{number}_depth_high.csv')
            
                # Save
            np.savetxt(low_save_path, dpt_low, delimiter=",")
            np.savetxt(med_save_path, dpt_med, delimiter=",")
            np.savetxt(high_save_path, dpt_high, delimiter=",")
            

            print(f"Saved Depth Splitted Data of {number}_depth.")
print("All Depth file Split Finished, 3 types of files saved.")


