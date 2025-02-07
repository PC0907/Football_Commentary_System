import cv2
import argparse
import streamlit as st
import os

# Streamlit configuration and custom CSS
st.set_page_config(page_title="Football Video Display", layout="wide")

# Set up argument parsing for input video name
parser = argparse.ArgumentParser(description="Display input and output videos side by side.")
parser.add_argument('input_video_name', type=str, help='Name of the input video file.')
args = parser.parse_args()

# Define paths
input_directory = '../Football-object-detection/test_videos/'
output_directory = '../Football-object-detection/output/'
input_video_path = os.path.join(input_directory, args.input_video_name)
output_video_path = os.path.join(output_directory, args.input_video_name.replace('.mp4', '_out.mp4'))

# Debugging: Print the paths to verify
st.write(f"Input video path: {input_video_path}")
st.write(f"Output video path: {output_video_path}")

# Check if input and output video paths exist
if not os.path.exists(input_video_path):
    st.error(f"Input video not found: {input_video_path}")
else:
    st.success(f"Input video found: {input_video_path}")

if not os.path.exists(output_video_path):
    st.error(f"Output video not found: {output_video_path}")
else:
    st.success(f"Output video found: {output_video_path}")




st.markdown("<h1 style='text-align: center; color: #2C3E50;'>Football Commentary Video Display</h1>", unsafe_allow_html=True)

st.markdown("""
    <style>
    .container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        margin-top: 20px;
    }
    .video-box {
        width: 48%;
        position: relative;
    }
    .video-box video {
        width: 100%;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
    }
    .controls {
        display: flex;
        justify-content: center;
        margin-top: 10px;
    }
    .button, .slider {
        font-size: 1rem;
        padding: 10px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin: 5px;
    }
    .slider {
        width: 80%;
    }
    </style>
    """, unsafe_allow_html=True)

# JavaScript to control playback and synchronization
st.markdown("""
    <script>
    const playPause = () => {
        const inputVideo = document.getElementById('inputVideo');
        const outputVideo = document.getElementById('outputVideo');
        if (inputVideo.paused) {
            inputVideo.play();
            outputVideo.play();
        } else {
            inputVideo.pause();
            outputVideo.pause();
        }
    };
    const syncVideos = () => {
        const inputVideo = document.getElementById('inputVideo');
        const outputVideo = document.getElementById('outputVideo');
        outputVideo.currentTime = inputVideo.currentTime;
    };
    </script>
    """, unsafe_allow_html=True)

# HTML layout for video display with controls
st.markdown(f"""
    <div class="container">
        <div class="video-box">
            <video id="inputVideo" controls muted ontimeupdate="syncVideos()">
                <source src="{input_video_path}" type="video/mp4">
            </video>
        </div>
        <div class="video-box">
            <video id="outputVideo" controls muted>
                <source src="{output_video_path}" type="video/mp4">
            </video>
        </div>
    </div>
    <div class="controls">
        <button class="button" onclick="playPause()">Play/Pause</button>
    </div>
""", unsafe_allow_html=True)



