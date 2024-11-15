# Import necessary libraries
import cv2  # OpenCV for handling video files
import argparse  # For handling command-line arguments
import streamlit as st  # Streamlit for web-based UI

# Set up argument parsing for input and output video paths
parser = argparse.ArgumentParser(description="Display input and output videos side by side.")
parser.add_argument('--input_video_path', type=str, default='../Football-object-detection/test_videos/',
                    help='Path to the input video directory.')
parser.add_argument('--output_video_path', type=str, default='../Football-object-detection/output/',
                    help='Path to the output video directory.')
args = parser.parse_args()  # Parse the command-line arguments

# Configure the Streamlit page
st.set_page_config(page_title="Football Video Display", layout="wide")
# Add a title to the web app, styled with color and alignment
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>Football Commentary Video Display</h1>", unsafe_allow_html=True)

# Define CSS for styling the video display area
st.markdown("""
    <style>
    .video-container {
        display: flex;  # Align videos horizontally
        justify-content: space-around;  # Space videos evenly across the container
        margin-top: 20px;
    }
    video {
        width: 45%;  # Set each video to take up 45% of the container width
        height: auto;
        border-radius: 10px;  # Add rounded corners
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);  # Add subtle shadow for depth
    }
    </style>
    """, unsafe_allow_html=True)

# Function to display videos side-by-side
def display_videos(input_path, output_path):
    # Open the video files at the provided paths
    input_cap = cv2.VideoCapture(input_path)  # Capture input video
    output_cap = cv2.VideoCapture(output_path)  # Capture output video

    # Create a Streamlit container to update frames in real time
    frame_container = st.empty()  # `st.empty()` is a placeholder that will update dynamically
    while input_cap.isOpened() and output_cap.isOpened():  # Continue as long as both videos are open
        ret_input, input_frame = input_cap.read()  # Read the next frame of the input video
        ret_output, output_frame = output_cap.read()  # Read the next frame of the output video

        # Stop if there are no more frames in either video
        if not ret_input or not ret_output:
            break

        # Convert frames from BGR to RGB color space (required by Streamlit for accurate color display)
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

        # Display both frames in the same container, side by side
        with frame_container:
            # Show frames as images in Streamlit, resizing to fit in the UI
            st.image([input_frame, output_frame], caption=["Input Video", "Output Video"], width=700)

    # Release video captures to free resources
    input_cap.release()
    output_cap.release()

# Call the display function using the command-line argument paths
display_videos(args.input_video_path, args.output_video_path)

