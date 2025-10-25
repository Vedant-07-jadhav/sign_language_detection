import pandas as pd
import os
import shutil

# Read cleaned data
df = pd.read_csv('Data/data_cleaned.csv')

# Split into train and test
df_train = df[df['split'] == 'train']
df_test = df[df['split'] == 'test']

# Get video IDs
train_video_ids = df_train['video_id'].tolist()
test_video_ids = df_test['video_id'].tolist()

# Load gloss encoding
df_coded_gloss = pd.read_csv('Data/wlasl_class_list.txt', sep='\t', header=None, names=['code', 'gloss'])
coded_dict = dict(zip(df_coded_gloss['code'], df_coded_gloss['gloss']))

print(f"Train videos: {len(train_video_ids)}")
print(f"Test videos: {len(test_video_ids)}")

# Define paths
video_source = 'Data/processed_videos'
output = 'data'

videos_path_train = os.path.join(output, 'videos', 'train')
videos_path_test = os.path.join(output, 'videos', 'test')

labels_path_train = os.path.join(output, 'labels', 'train.txt')
labels_path_test = os.path.join(output, 'labels', 'test.txt')

# Create directories
os.makedirs(videos_path_train, exist_ok=True)
os.makedirs(videos_path_test, exist_ok=True)
os.makedirs(os.path.dirname(labels_path_train), exist_ok=True)

# Function to move files and create labels
def move_files_and_create_labels(df, video_ids, video_source, labels_path, video_dest):
    labels = []
    moved = 0
    missing = 0
    
    for video_id in video_ids:
        video_id_str = str(video_id).zfill(5)
        video_file = f"{video_id_str}.mp4"
        video_path = os.path.join(video_source, video_file)
        
        if os.path.exists(video_path):
            dest_path = os.path.join(video_dest, video_file)
            shutil.copy2(video_path, dest_path)
            
            gloss_code = df.loc[df['video_id'] == video_id, 'gloss_encode'].values[0]
            labels.append(f"{video_file} {gloss_code}")
            moved += 1
        else:
            missing += 1
    
    # Write labels
    with open(labels_path, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    
    return moved, missing

# Move train files
train_moved, train_missing = move_files_and_create_labels(
    df_train, train_video_ids, video_source, labels_path_train, videos_path_train
)

# Move test files
test_moved, test_missing = move_files_and_create_labels(
    df_test, test_video_ids, video_source, labels_path_test, videos_path_test
)

print(f"\nTrain: {train_moved} moved, {train_missing} missing")
print(f"Test: {test_moved} moved, {test_missing} missing")
print("Done!")