import cv2 
import numpy as np
import matplotlib.pyplot as plt 
import sys 
import os 
import time


def capture_image(save_path='', camera=0, image_name='', image_format="png"):
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
        The format of the image

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
