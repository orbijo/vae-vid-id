from pytubefix import YouTube
import pandas as pd
import os
import json
import subprocess  # Use ffmpeg directly
import re

# CONFIG
num_samples = 50  # Number of videos to process
filename = "panda70m_training_2m.csv"  # Dataset CSV filename
directory = "data/"  # CSV directory

# SETUP
csv = os.path.join(directory, filename)
base_df = pd.read_csv(csv)
df = base_df[['videoID', 'url', 'timestamp', 'caption']].head(num_samples)  # Select relevant columns

# Folder for output (Following paper directory structure)
base_dir_vid = "examples/videos"
gt_dir_vid = os.path.join(base_dir_vid, "gt")
downloads_dir = "downloads"
os.makedirs(gt_dir_vid, exist_ok=True)
os.makedirs(downloads_dir, exist_ok=True)

# Function to download a YouTube video
def download_youtube_video(url, save_path):
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
        if stream:
            output_dir = os.path.dirname(save_path)
            filename = os.path.basename(save_path)
            stream.download(output_path=output_dir, filename=filename)
            correct_path = os.path.join(output_dir, filename)
            print(f"Downloaded video to {correct_path}")
            return correct_path
        else:
            print(f"No suitable stream found for {url}")
            return None
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

# Function to convert 'HH:MM:SS.SSS' to seconds
def time_to_seconds(time_str):
    match = re.match(r"(?:(\d+):)?(\d+):(\d+\.\d+)", time_str)
    if match:
        hours = float(match.group(1)) if match.group(1) else 0
        minutes = float(match.group(2))
        seconds = float(match.group(3))
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid time format: {time_str}")
    
# Function to extract video clips using ffmpeg
def extract_clip_ffmpeg(input_video, start_time, end_time, output_clip):
    try:
        # Convert 'HH:MM:SS.SSS' format to seconds
        start_time_sec = time_to_seconds(start_time)
        end_time_sec = time_to_seconds(end_time)

        command = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-ss", str(start_time_sec),
            "-to", str(end_time_sec),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-c:a", "aac",
            "-strict", "experimental",
            output_clip
        ]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Debugging output
        # print("FFmpeg Output:", result.stdout)
        # print("FFmpeg Error:", result.stderr)

        # if result.returncode != 0:
        #     print(f"❌ Error extracting clip {output_clip}")
        # else:
        #     print(f"✅ Extracted clip: {output_clip}")

    except Exception as e:
        print(f"Error in extract_clip_ffmpeg: {e}")


# Start video numbering from 1
video_index = 1  

# Process each row in the DataFrame
for _, row in df.iterrows():
    video_id = row["videoID"]
    url = row["url"]
    timestamps = eval(row["timestamp"])  # Convert string list to actual list
    captions = eval(row["caption"])  

    print(f'Downloading video: ID:{video_id}, URL:{url}')

    # Define video path for the full YouTube video
    full_video_path = os.path.join(downloads_dir, f"full_{video_id}.mp4")

    # Download the video if not already present
    if not os.path.exists(full_video_path):
        downloaded_video = download_youtube_video(url, full_video_path)
        if not downloaded_video:
            continue  # Skip if download fails

    # Process each timestamp to extract clips
    for timestamp, caption in zip(timestamps, captions):
        start_time, end_time = timestamp  

        # Define output paths
        clip_path = os.path.join(gt_dir_vid, f"video{video_index}.mp4")
        annotation_path = os.path.join(gt_dir_vid, f"video{video_index}.txt")

        # Extract video segment using improved ffmpeg method
        extract_clip_ffmpeg(full_video_path, start_time, end_time, clip_path)

        # Save the corresponding annotation
        with open(annotation_path, "w") as f:
            f.write(caption)
        print(f"Saved annotation: {annotation_path}")

        # Increment the video index for the next clip
        video_index += 1  
