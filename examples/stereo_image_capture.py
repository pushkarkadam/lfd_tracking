import cv2 
import numpy as np 


cap = cv2.VideoCapture(4)

if cap.isOpened() == 0:
    exit(-1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


counter = 0

print("Initialising camera...")

while True:
    print(f"Frame Count: {counter}")

    retval, frame = cap.read()

    counter += 1

    if counter > 5:
        break



print(retval)

cv2.imwrite("image.png", frame)

cap.release()