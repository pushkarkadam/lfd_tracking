import numpy as np
import cv2 as cv
import sys 
import time
import argparse

# command line arguments
parser = argparse.ArgumentParser(description='Pass arguments')

parser.add_argument("-o", "--output_video", default=f'{int(time.time())}', help="Video output name")
parser.add_argument("-c", "--camera", default=0, help="Specify the camera. Defaults to 0.")
parser.add_argument('-m', "--mirror", default=False, help="Mirrors the camera")

args = parser.parse_args()

# Assigning the command line arguments
video_name = f'{args.output_video}.avi' 
camera_number = int(args.camera)
mirror = args.mirror

# video capture object
try:
    cap = cv.VideoCapture(camera_number)
except Exception as e:
    print(e)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')

out = cv.VideoWriter(video_name, fourcc, 20.0, (640,  480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if mirror:
        frame = cv.flip(frame, 1)
    
    # write the flipped frame
    out.write(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()