import cv2 
import numpy as np
import matplotlib.pyplot as plt 
import sys 
import os 
import time
from tqdm import tqdm 


def capture_image(save_path='', camera=0, image_name='', image_format="png", resolution=(720, 480), counter=5):
    """Captures an image.
    
    Parameters
    ----------
    save_path: str, optional
        A path to store the data.
    camera: int, optional
        The webcamera input.
    image_name: str, optional
        The name of the image.
    format: str, optional, default ``"png"``
        The format of the image.
    resolution: tuple, default ``(720, 480)``
        Image resolution to capture the image. Use ``(2560, 720)`` resolution for stereo image capture.
    counter= int, default ``5``.
        Counts the iteration before taking the image.
        This allows the camera to have enough time to get the input and avoid the green image that results from initial switching of the camera.

    Returns
    -------
    numpy.ndarray
        A numpy array of 3 channels of image.

    Examples
    --------
    >>> from lfdtrack import *
    >>> I = capture_image()

    """

    cam = cv2.VideoCapture(camera)

    if cam.isOpened() == 0:
        exit(-1)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    print("Initialising Camera...")

    print("Testing camera...")

    for i in tqdm(list(range(0,counter))):
        result, image = cam.read()

    if result:
        # writing the image
        if save_path:
            # check if the image name is given as input
            if image_name:
                image_name_format = image_name + '.' + image_format
            # Use a timestamp to name the image when the image name is not given
            else:
                image_name_format = str(int(time.time())) + '.' + image_format

            # Combining the name of the image with the save path
            image_path = os.path.join(save_path, image_name_format)

            cv2.imwrite(image_path, image)
            print(f"Image: {image_name_format} successfully saved at {save_path}.")
    else:
        print("No image detected. Please try again.")

    return image


def stereo_split(I, sections=2, axis=1):
    """Returns a tuple of left and right image of the stereo image.

    Parameters
    ----------
    I: numpy.ndarray
        A stereo image which is located next to each other.
    sections: int, default ``2``
        Number of sections to split the image.
    axis: int, default ``1``
        Axis about which to split.
        ``1`` used for splitting around y-axis.
        ``0`` used for splitting around x-axis.
    
    Returns
    -------
    tuple
        A tuple of split image.

    Examples
    --------
    >>> from lfdtrack import *
    >>> I = np.zeros((500, 1000))
    >>> imgL, imgR = stereo_split(I)
    >>> imgL.shape
    (500, 500)
    
    """

    img_split = np.split(I, indices_or_sections=sections, axis=axis)

    return img_split 

