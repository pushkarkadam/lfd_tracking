import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression


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
    >>> from lfdtrack.fingertrack import *
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
    >>> from lfdtrack.fingertrack import *
    >>> video_path = "~/path/to/video/file.mp4"
    >>> frames = get_video_frames(video_path)
    >>> rgb_frames = convert_frames(frames, cv2.COLOR_BGR2RGB)
    
    """
    
    converted_frames = []
    
    for frame in frames:
        converted_frames.append(cv2.cvtColor(frame, color_conversion_code))
        
    return converted_frames

def finger_tracking(frames, 
                    hand_landmark, 
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    verbose=False
                   ):
    """Returns the dict of co-ordinates that maps axes to co-ordinates.
    
    The finger tip co-ordinates (x, y, z) are extracted and from
    each keyframe and stored in a dictionary in the form of list.
    
    The x and y co-ordinates are multiplied by the image size
    that helps in tracking the points relative to the image frame
    where the finger tip was detected.
    
    The same is not true for z co-ordinates. Since there is no depth
    information available, the exact location cannot be determined
    in 3-D space.
    
    Parameters
    ----------
    frames: list
        A list of image frames from a video as numpy array.
    hand_landmark: str
        Hand landmark that identifies different location on the hands.
        Each of the hand landmark is identified by a unique name and number assigned to it.
        Check out `mediapipe docs`_ for more information.
        
        .. _mediapipe docs: https://google.github.io/mediapipe/solutions/hands#hand-landmark-model
    max_num_hands: int, default ``2``
        The number of hands to be detected.
        The observation is such that the number of data points is proportional to the ``max_num_hands``.
    min_detection_confidence: float, default ``0.5``
        The minimum confidence required to declare the hand is present or not in the frame.
        If the hand is not completely visible in the video, having a lower confidence such as ``0``
        would give better results.
    verbose: bool, optional, defaul ``False``
        Allows to print the hand landmark results in the console.
        
    Returns
    -------
    dict
        A dictionary that maps co-ordinate type to the list of co-ordinates.
    
    Examples
    --------
    >>> from lfdtrack.fingertrack import *
    >>> video_path = "~/path/to/video/file.mp4"
    >>> frames = get_video_frames(video_path)
    >>> rgb_frames = convert_frames(frames, cv2.COLOR_BGR2RGB)
    >>> fingertips = finger_tracking(rgb_frames, hand_landmark='INDEX_FINGER_TIP', max_num_hands=2, min_detection_confidence=0, verbose=False)

    """
    fingertip = dict()
    fingertip['x'] = []
    fingertip['y'] = []
    fingertip['z'] = []
    
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence
    ) as hands:
        for idx, image in enumerate(frames):
            results = hands.process(image)
            if verbose:
                print('Handedness:', results.multi_handedness)
                
            if not results.multi_hand_landmarks:
                continue
                
            image_height, image_width, _ = image.shape
            
            annotated_image = image.copy()
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Updating the fingertip location
                fingertip['x'].append(hand_landmarks.landmark[getattr(mp_hands.HandLandmark, hand_landmark)].x * image_width)
                fingertip['y'].append(hand_landmarks.landmark[getattr(mp_hands.HandLandmark, hand_landmark)].y * image_height)
                fingertip['z'].append(hand_landmarks.landmark[getattr(mp_hands.HandLandmark, hand_landmark)].z) 
    
    return fingertip

def fit_track(x, y):
    """Fits the data and returns a list of predictive values.
    
    The data points from the fingertip tracking from the
    mediapipe is not in a straight line.
    Using linear regression to fit the data on a straight line.
    
    Parameters
    ----------
    x: list
        A list of x co-ordinates.
    y: list
        A list of y co-ordinates.
        
    Returns
    -------
    list
        A list of predicted y values.
        
    Examples
    --------
    >>> from lfdtrack.fingertrack import fit_track
    >>> x = [1,2,3,4]
    >>> y = [1,4,7,6]
    >>> y_pred = fit_track(x, y)
    
    """
    x = np.array(x).reshape((-1, 1))
    y = np.array(y)
    
    model = LinearRegression()
    
    model.fit(x, y)
    
    y_pred = model.predict(x)
    
    return list(y_pred)

def region_box_coords(image, x_vals, y_vals, scaling_factor=100):
    """Returns the region box co-ordinates.
    
    Finds the maximum and minimum x and y co-ordinates
    where the line was traced in the image.
    
    Parameters
    ----------
    image: numpy.ndarray
        An RGB image.
    x_vals: list
        A list of all the x co-ordinates where the finger
        was tracked.
    y_vals: list
        A list of all the y co-ordinates where the finger
        was tracked.
    scaling_factor: int, default ``100``
        The scaling factor is the value in pixel.
        It denotes how further away the padding is 
        required for the image.
        
    Returns
    -------
    tuple
        A tuple of int values in the
        order (x_min, y_min, x_max, y_max)
    
    Examples
    --------
    >>> from lfdtrack.fingertrack import *
    >>> import numpy as np
    >>> I = np.identity(5)
    >>> x = list(range(1,3))
    >>> y = list(range(1,3))
    >>> boxes = point_box(I, x, y, scaling_factor=100)
    
    """
    # Getting image dimensions
    img_height = image.shape[0]
    img_width = image.shape[1]
    
    x_pad = []
    y_pad = []
    
    # copying for smaller named variable
    sf = scaling_factor
    
    for x, y in zip(x_vals, y_vals):
        # x padding
        x1 = x-sf if x-sf > 0 else 0
        x2 = x+sf if x+sf < img_width else img_width
        
        # y padding
        y1 = y-sf if y-sf > 0 else 0
        y2 = y+sf if y-sf < img_height else img_height
              
        # Adding the scaling values to the padding list
        x_pad.append(x1)
        x_pad.append(x2)
        
        y_pad.append(y1)
        y_pad.append(y2)
        
    # Converting the min and max value to int
    x_min = int(min(x_pad))
    x_max = int(max(x_pad))
    y_min = int(min(y_pad))
    y_max = int(max(y_pad))
    
    return (x_min, y_min, x_max, y_max)

def point_box(image, x_vals, y_vals, scaling_factor=100):
    """Returns an array of points.
    
    Returns the box that has the length
    of twice the scaling_factor pixels.
    
    Parameters
    ----------
    image: numpy.ndarray
        An RGB image.
    x_vals: list
        A list of x co-ordinates.
    y_vals: list
        A list of y co-ordinates.
    scaling_factor: int
        Scaling the area vertically and horizontally
        for a point.
        
    Returns
    -------
    list
        A list of ``numpy.ndarray`` that consists of the
        end co-ordinates of the box around the pixel.
        
    Examples
    --------
    >>> from lfdtrack.fingertrack import *
    >>> import numpy as np
    >>> I = np.identity(5)
    >>> x = list(range(1,3))
    >>> y = list(range(1,3))
    >>> boxes = point_box(I, x, y, scaling_factor=100)
    
    """
    boxes = []
    
    for x, y in zip(x_vals, y_vals):
        x_min, y_min, x_max, y_max = region_box_coords(image, [x], [y], scaling_factor=scaling_factor)
        
        polygon = np.array([
            [(x_min, y_min), 
             (x_min, y_max),
             (x_max, y_max),
             (x_max, y_min)
            ]
        ])
        boxes.append(polygon)
        
    return boxes  

def region_of_interest(image, boxes):
    """Returns the image with region of interest.
    
    The image consists of only the region of interest
    with the original image while the rest of the 
    region is colored black with 0.
    
    Parameters
    ----------
    image: numpy.ndarray
        An RGB image.
    boxes: list
        A list of ``numpy.ndarray``.
        
    Returns
    -------
    numpy.ndarray
        An image with the region of interest.
        Use ``matplotlib.pyplot.imshow(image)``
        to see the image in Jupyter Notebook.
    
    Examples
    --------
    >>> from lfdtrack.fingertrack import *
    >>> import numpy as np
    >>> boxes = [np.array([[1,2],[3,4]]), np.array([[1,2],[3,4]])]
    >>> image = np.identity(3)
    >>> im_roi = region_of_interest(image, boxes)
    
    """

    mask = np.zeros_like(image)

    for box in boxes:
        image_poly = cv2.fillPoly(mask, box, color=(255,255,255))
        
    masked_image = cv2.bitwise_and(image, image_poly)
    
    return masked_image