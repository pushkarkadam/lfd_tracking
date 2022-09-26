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

def convert_frames(frames, color_conversion_code=cv2.COLOR_BGR2RGB):
    """Returns a list the frames converted to the specified type.
    
    The function iterates over the list of frames and returns the
    list of the frames that are all converted to the type mentioned.
    
    Parameters
    ----------
    frames: list
        A list of image frames from a video as numpy array.
    color_conversion_code: int, default ``cv2.COLOR_BGR2RGB``
        The ``color_conversion_code`` is of type ``int``.
        The default value uses the ``enum`` type defined in ``cv2`` module.
        Use the enum equivalent gives clear explanation of the 
        conversion code.
        The input might as well be a number ranging from 0 to 143 (inclusive).
        
        For more information on the color conversion code,
        see the `opencv docs`_.
        
        .._opencv docs: https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
    
    Returns
    -------
    list
        A list of converted image frames of the video.
        
    Examples
    --------
    >>> from lfdtrack.handrec import *
    >>> video_path = "~/path/to/video/file.mp4"
    >>> frames = get_video_frames(video_path)
    >>> rgb_frames = convert_frames(frames, cv2.COLOR_BGR2RGB)
    
    """
    
    converted_frames = []
    
    for frame in frames:
        converted_frames.append(cv2.cvtColor(frame, color_conversion_code))
        
    return converted_frames