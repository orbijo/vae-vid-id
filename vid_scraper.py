### Scraper for Panda70m Dataset ###
import yt_dlp
import pandas as pd
import os

# CSV Path
filename = "panda70m_training_2m.csv"
csv = "csv/" + filename
df = pd.read_csv(csv)

# Folder for output (Following paper directory structure)
# For videos
base_dir_vid = "examples/videos"
gt_dir_vid = os.path.join(base_dir_vid, "gt")  # Ground truth videos
recon_dir_vid = os.path.join(base_dir_vid, "recon")  # Reconstructed videos
os.makedirs(gt_dir_vid, exist_ok=True)
os.makedirs(recon_dir_vid, exist_ok=True)
print(f"Directories created:\n- {gt_dir_vid}\n- {recon_dir_vid}")

### DECIDED NOT TO USE IMAGES ###
# For images
# base_dir_img = "examples/videos"
# gt_dir_img = os.path.join(base_dir_img, "gt")  # Ground truth videos
# recon_dir_img = os.path.join(base_dir_img, "recon")  # Reconstructed videos
# os.makedirs(gt_dir_img, exist_ok=True)
# os.makedirs(recon_dir_img, exist_ok=True)
# print(f"Directories created:\n- {gt_dir_img}\n- {recon_dir_img}")

def get_samples(video_id, url, timestamp):
    video_path = os.path.join(gt_dir_vid, f"{video_id}.mp4")
    trimmed_path = os.path.join(gt_dir_vid, f"{video_id}_trimmed.mp4")

    if not os.path.exists(video_path):
        ydl_opts = {"outtmpl":video_path}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    
    start_time, end_time = timestamp[0]

    os.system(f"ffmpeg -i {video_path} -ss {start_time} -to {end_time} -c:v libx264 -c:a aac {trimmed_path}")

    return trimmed_path

# Process first 50 videos
for i, row in df.iterrows():
    video_id = row["videoID"]
    url = row["url"]
    timestamps = eval(row["timestamp"])  # Convert string list to actual list
    trimmed_video = get_samples(video_id, url, timestamps)

    print(f"Processed {trimmed_video}")
    if i >= 50:
        break