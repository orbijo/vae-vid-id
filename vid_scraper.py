### Scraper for Panda70m Dataset and Format it according to paper documentation (Vn.mp4 and Vn.txt (annotation))###
from pytubefix import YouTube
import pandas as pd
import os
import json

# CONFIG
num_samples = 1 # number of samples
filename = "panda70m_training_2m.csv" # data set csv filename
directory = "data/" # csv directory

# SETUP
csv = directory + filename
base_df = pd.read_csv(csv)
df = base_df[['videoID', 'url', 'timestamp', 'caption']].head(num_samples) # Select only relevant columns

# Folder for output (Following paper directory structure)
base_dir_vid = "examples/videos"
gt_dir_vid = os.path.join(base_dir_vid, "gt")
recon_dir_vid = os.path.join(base_dir_vid, "recon")
os.makedirs(gt_dir_vid, exist_ok=True)
os.makedirs(recon_dir_vid, exist_ok=True)

def get_samples(video_id, url, timestamps, video_name):
    yt = YouTube(url)
    print(yt.title)

    # video_path = os.path.join(gt_dir_vid, f"{video_id}.mp4")
    # trimmed_path = os.path.join(gt_dir_vid, f"{video_id}_trimmed.mp4")

    # if not os.path.exists(video_path):
    #     ydl_opts = {"outtmpl":video_path}
    #     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    #         ydl.download([url])
    
    # start_time, end_time = timestamp[0]

    # os.system(f"ffmpeg -i {video_path} -ss {start_time} -to {end_time} -c:v libx264 -c:a aac {trimmed_path}")

    # return trimmed_path

for i, row in df.iterrows():
    video_name = f"video{i+1}" # filename: video1.mp4, video1.txt, video2.mp4 ... etc
    video_id = row["videoID"]
    url = row["url"]
    timestamps = eval(row["timestamp"]) # timestamp in csv is a str list. convert to actual list
    get_samples(video_id, url, timestamps, video_name)