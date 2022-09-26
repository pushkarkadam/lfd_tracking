import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('..')

from lfdtrack.fingertrack import *


video_path = '../data/videos/finger_track0.mp4'

# Breaking down the video into a list of images of numpy type
frames = get_video_frames(video_path)

# Converting all the frames from the video to RGB
rgb_frames = convert_frames(frames, cv2.COLOR_BGR2RGB)

# Collectig the co-ordinates (x, y, z) from the video for index finger
fingertips = finger_tracking(rgb_frames, 
                             hand_landmark='INDEX_FINGER_TIP', 
                             max_num_hands=2, 
                             min_detection_confidence=0, 
                             verbose=False)

# Using Linear regression to fit the data on a stright line
y_pred = fit_track(fingertips['x'], fingertips['y'])

implot = plt.imshow(rgb_frames[-1])

plt.plot(fingertips['x'], y_pred, linestyle='solid', color='r')

# Uncomment the following line for saving the figure
# plt.savefig('reg_line', format='svg', dpi=100)
plt.show()