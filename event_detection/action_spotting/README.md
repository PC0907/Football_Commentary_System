# Guide to Event Detection 

## Setup
1) Clone the appropriate repository
   ```sh
   git clone !git clone https://github.com/lRomul/ball-action-spotting
   cd ball-action-spotting
2) Install the required libraries
   ```sh
   pip install pytorch-argus opencv-python scipy tqdm kornia
3) Run the script using the following line:
   ```sh
   python predictions.py --model_path (enter model path) --video_path (enter video path) --output_path (enter path for .jsons file - output predictions) --half 1
