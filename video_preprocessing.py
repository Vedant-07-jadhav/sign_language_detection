import subprocess
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
import cv2

def process_with_ffmpeg(video_id, bbox):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    input_path = f'Data/videos/{video_id}.mp4'
    output_path = f'Data/processed_videos/{video_id}.mp4'
    
    os.makedirs('Data/processed_videos/', exist_ok=True)
    
    # FFmpeg command for crop + grayscale conversion
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vf', f'crop={width}:{height}:{x1}:{y1},format=gray',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
        '-loglevel', 'error',
        '-y',
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        
        # Verify output file was created and has size > 0
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return video_id, True, None
        else:
            return video_id, False, "Output file not created or empty"
            
    except subprocess.TimeoutExpired:
        return video_id, False, "Processing timeout (>5 minutes)"
    except subprocess.CalledProcessError as e:
        return video_id, False, str(e.stderr)
    except FileNotFoundError:
        return video_id, False, "FFmpeg not installed"
    except Exception as e:
        return video_id, False, str(e)

def process_wrapper(args):
    video_id, bbox_str = args
    try:
        bbox = [int(x.strip()) for x in bbox_str.strip('[]').split(',')]
        return process_with_ffmpeg(video_id, bbox)
    except Exception as e:
        return video_id, False, f"Bbox parsing error: {str(e)}"

if __name__ == '__main__':
    # Read original CSV
    df = pd.read_csv('Data/data.csv')
    df['video_id'] = df['video_id'].astype(str).str.zfill(5)
    
    print(f"Total videos in dataset: {len(df)}")
    
    video_args = list(zip(df["video_id"], df["bbox"]))
    
    num_workers = min(multiprocessing.cpu_count() * 2, 16)
    
    print(f"Processing {len(video_args)} videos using {num_workers} workers with FFmpeg...")
    
    # Process videos in parallel with progress bar
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_wrapper, video_args), total=len(video_args)))
    
    # Separate successful and failed videos
    successful_videos = [vid for vid, success, _ in results if success]
    failed_videos = [(vid, err) for vid, success, err in results if not success]
    
    print(f"\n{'='*60}")
    print(f"Processing Summary:")
    print(f"{'='*60}")
    print(f"Total videos: {len(video_args)}")
    print(f"Successfully processed: {len(successful_videos)}")
    print(f"Failed: {len(failed_videos)}")
    print(f"Success rate: {len(successful_videos)/len(video_args)*100:.2f}%")
    
    # Save failed videos log
    if failed_videos:
        print(f"\n{'='*60}")
        print(f"Failed Videos (first 10):")
        print(f"{'='*60}")
        for vid, err in failed_videos[:10]:
            print(f"  {vid}: {err}")
        
        # Save complete list of failed videos
        failed_df = pd.DataFrame(failed_videos, columns=['video_id', 'error'])
        failed_df.to_csv('Data/failed_videos.csv', index=False)
        print(f"\nComplete list of failed videos saved to 'Data/failed_videos.csv'")
    
    # Filter dataframe to keep only successful videos
    df_cleaned = df[df['video_id'].isin(successful_videos)].copy()
    
    print(f"\n{'='*60}")
    print(f"Cleaning Dataset:")
    print(f"{'='*60}")
    print(f"Original dataset size: {len(df)}")
    print(f"Cleaned dataset size: {len(df_cleaned)}")
    print(f"Removed: {len(df) - len(df_cleaned)} videos")
    
    # Save cleaned CSV
    df_cleaned.to_csv('Data/data_cleaned.csv', index=False)
    print(f"\nCleaned dataset saved to 'Data/data_cleaned.csv'")
    
    # Optional: Delete failed processed videos (if any were partially created)
    print(f"\n{'='*60}")
    print(f"Cleaning up failed output files...")
    print(f"{'='*60}")
    deleted_count = 0
    for vid, _ in failed_videos:
        failed_output = f'Data/processed_videos/{vid}.mp4'
        if os.path.exists(failed_output):
            try:
                os.remove(failed_output)
                deleted_count += 1
            except Exception as e:
                print(f"Could not delete {failed_output}: {e}")
    
    if deleted_count > 0:
        print(f"Deleted {deleted_count} failed/incomplete output files")
    
    print(f"\n{'='*60}")
    print(f"All done! Use 'Data/data_cleaned.csv' for training your model")
    print(f"{'='*60}")
    
    def verify_video(video_id):
        """Check if video can be opened and read"""
        video_path = f'Data/videos/{video_id}.mp4'
        
        if not os.path.exists(video_path):
            return video_id, False, "File not found"
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return video_id, False, "Cannot open video"
            
            # Try to read first frame
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return video_id, False, "Cannot read frames"
            
            # Check video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            if fps == 0 or frame_count == 0:
                cap.release()
                return video_id, False, "Invalid video properties"
            
            cap.release()
            return video_id, True, None
            
        except Exception as e:
            return video_id, False, str(e)

# Add this before processing:
print("Verifying video files...")
verify_args = df["video_id"].tolist()
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    verify_results = list(tqdm(executor.map(verify_video, verify_args), total=len(verify_args)))

valid_videos = [vid for vid, valid, _ in verify_results if valid]
invalid_videos = [(vid, err) for vid, valid, err in verify_results if not valid]

print(f"Valid videos: {len(valid_videos)}/{len(df)}")
if invalid_videos:
    print(f"Invalid videos found: {len(invalid_videos)}")
    # Save and remove invalid videos from processing
    df = df[df['video_id'].isin(valid_videos)]