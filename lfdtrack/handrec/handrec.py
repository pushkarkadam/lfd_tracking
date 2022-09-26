import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def get_video_frames(video_path):
    """Returns a list of image frames as numpy array.
    
    The frames are extracted from the video and each frame
    is an image that is represented as a numpy array.
    
    Paramaters
    ----------
    video_path: str
        The path to the video file.
        
    Returns
    -------
    list
        A list of image frames in the video as numpy array.
    
    Examples
    --------
    >>> from lfdtrack.handrec import *
    >>> video_path = "~/path/to/video/file.mp4"
    >>> frames = get_video_frames(video_path)
    
    """
    # Empty list to store the frames
    frames = []
    
    video = cv2.VideoCapture(video_path)
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(total_frames):
        _, frame = video.read()
        # Appending the frame to the list
        frames.append(frame)
        
    return frames