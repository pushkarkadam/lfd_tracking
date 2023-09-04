import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import json
import os
import time
from tqdm import tqdm

sys.path.append('../')

from lfdtrack import *


# Importing YOLO weights
model_path = '../models/freiHand0.pt'

model = YOLO(model_path)

video_path = '../data/videos/left_hand_weldless.avi'

video_frames = get_video_frames(video_path)

# Flip video along y axis
flipped_frames = []
flip_type=1

for frame in video_frames:
    flipped_image = cv2.flip(frame, flip_type)
    flipped_frames.append(flipped_image)
    del flipped_image

# Pose detection
render = YOLOHandPose(frames=flipped_frames)

render.process()

render.render_pose(font_scale=0.5)

# Uncomment the following line to save
# This will appear as if this is a right hand in the video.
# render.save(filename='left_flipped')

# Flipping pose results
h, w, _ = flipped_frames[0].shape

T = np.array([[-1, 0, 0], 
              [0, 1, 0],  
              [w, 0, 1]])

flipped_lmks = flip_pose(render.xy, T)

# Creating a new object for YOLOHandPose with original video of left hand
render2 = YOLOHandPose(frames=video_frames)

# Assigning the flipped landmarks to the xy attribute
render2.xy = flipped_lmks

# Rendering the pose on the original left hand video with flipped coordinates
render2.render_pose(font_scale=0.5)

# Uncomment the following line to save
# render2.save(filename='flipped_rendered_left')