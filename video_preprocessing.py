import cv2 
import numpy as np
import os
import pandas as pd

def crop_videos_convert_Gray_scale(video_id, bbox, f_start=1, f_end=-1, fps=0):
    video_path = os.path.join('Data/videos/', f'{video_id}.mp4')
    
    x1, y1, x2, y2 = bbox;
    cmap = cv2.VideoCapture(video_path)
    if not cmap.isOpened():
        print(f'could not open {video_id}.mp4 file')
        exit()
    
    fps = cmap.get(cv2.CAP_PROP_FPS)
    width = int(cmap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cmap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if f_end == -1:
        f_end = int(cmap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    start_frame= f_start
    end_rrame = f_end
    output_dir = 'Data/processed_videos/'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join('Data/processed_videos/', f'{video_id}.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Choose a suitable codec. 'mp4v' or 'XVID' are common.

    cropped_width = np.abs(x2 - x1)
    cropped_height = np.abs(y2 - y1)
    out = cv2.VideoWriter(output_file, fourcc, fps, (cropped_width, cropped_height), isColor=False)

    frame_count = 0
    while True:
        ret, frame = cmap.read()
        
        if not ret:
            print(f"Error: Unable to read the video frame of {video_id}.")
            break
        if start_frame<= frame_count<=end_rrame:
            cropped_frame = frame[y1:y2, x1:x2]
            gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            out.write(gray_frame)
        frame_count+=1
        if frame_count> end_rrame:
            
            break
    print("done")
    cmap.release()
    out.release()
    cv2.destroyAllWindows()


df = pd.read_csv('Data/data.csv')
df['video_id'] = df['video_id'].astype(str).str.zfill(5)
dict = dict(zip(df["video_id"], df["bbox"]))
for video_id, bbox in dict.items():
    bbox = bbox.strip('[]')
    bbox = bbox.split(',')
    bbox = [int(x.strip()) for x in bbox]
    crop_videos_convert_Gray_scale(bbox=bbox, video_id=video_id)
    
print("Completed preprocessing")
    
    
