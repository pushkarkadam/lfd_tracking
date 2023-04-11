import numpy as np
import cv2
import sys 
import time
import argparse
import os

# command line arguments
parser = argparse.ArgumentParser(description='Pass arguments')

parser.add_argument("-o", "--output_video", default='', help="Video output name.")
parser.add_argument("-c", "--camera", default=0, help="Specify the camera. Defaults to 0.")
parser.add_argument('-m', "--mirror", default=False, help="Mirrors the camera.")
parser.add_argument('-p', "--path", default="", help="path to store the video footage.")
parser.add_argument('-width', "--width", default=1280, help="Width of the image resolution. Defaults to 640.")
parser.add_argument('-height', "--height", default=720, help="Height of the image resolution. Defaults to 420.")

args = parser.parse_args()


resolution = (int(args.width), int(args.height))

# Assigning the command line arguments
if args.output_video == '':
    output_video = f'{int(time.time())}.avi'
else:
    output_video = f'{args.output_video}.avi' 
camera_number = int(args.camera)
mirror = args.mirror

video_name = os.path.join(args.path, output_video)

# video capture object
try:
    cap = cv2.VideoCapture(camera_number)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
except Exception as e:
    print(e)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter(video_name, fourcc, 20.0, (resolution[0],  resolution[1]))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if mirror:
        frame = cv2.flip(frame, 1)
    
    # write the flipped frame
    try:
        out.write(frame)
    except Exception as e:
        print("Make sure the path entered is correct!")
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()