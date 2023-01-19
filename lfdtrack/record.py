import numpy as np
import cv2 as cv
import sys 
import time

if len(sys.argv) > 1:
    video_name = f'{sys.argv[1]}.avi'
    
else:
    video_name = f'video_{int(time.time())}.avi'
    cam_number = 0
    print('Usage: python record.py <video_name> <cam_number>')

if len(sys.argv) == 3:
    cam_number = int(sys.argv[2])
else:
    cam_number = 0
    print('Using default web camera')


# video capture object
try:
    cap = cv.VideoCapture(cam_number)
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
    # frame = cv.flip(frame, 0)
    # write the flipped frame
    out.write(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()