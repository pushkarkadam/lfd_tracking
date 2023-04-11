import cv2 
import numpy as np 


def weldment_roi(image, 
                 blur_kernel_size=11,
                 binarization_max_value=255,
                 binarization_adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                 binarization_threshold_type=cv2.THRESH_BINARY_INV,
                 binarization_block_size=11,
                 binarization_constant=2,
                 dilation_kernel_size=(3,3),
                 dilation_iterations=1,
                 erosion_kernel_size=(3,3),
                 erosion_iterations=1,
                 erosion_border_type=cv2.BORDER_REFLECT,
                 contour_mode=cv2.RETR_TREE,
                 contour_method=cv2.CHAIN_APPROX_NONE
                ):
    """Returns an image with isolated ROI of the weldment.
    
    Parameters
    ----------
    image: numpy.ndarray
        A RGB color image.
    blur_kernel_size: int, default ``11``
        Kernel size for median blur filter (values from 3, 5, 7, 9 and so on)
    binarization_max_value: int, default ``255``
        Maximum value for binarization
    binarization_adaptive_method, int, default ``cv2.ADAPTIVE_THRESH_GAUSSIAN_C``,
        Adaptive binarization method.
        Using the constant from opencv and input.
        Input can be an ``int`` corresponding to the adaptiv threshold value.
    binarization_threshold_type: int, default ``cv2.THRESH_BINARY_INV``
        Binarization threshold type for converting image to binary.
        Also inverts the binary image.
        For not inverting the image use ``cv2.THRESH_BINARY``
    binarization_block_size: int, default ``11``
        Block size for binarization process.
    binarization_constant: int, default ``2``
        Binarization constant
    dilation_kernel_size: tuple, default ``(3,3)``
        Kernel size for dilation operation.
        Provide a tuple of odd numbers.
    dilation_iterations: int, default ``1``
        Number of times to perform dilation
    erosion_kernel_size: tuple, default ``(3,3)``
        Size of the kernel for erosion
    erosion_iterations: int, default ``1``
        Number of times to perform the iterations.
    erosion_border_type: int default ``cv2.BORDER_REFLECT``
        The type of border to use while eroding.
        The input is used as per the opencv map of method to integer value.
    contour_mode: int, default ``cv2.RETR_TREE``
        Contour mode as per the opencv map to integer value.
    contour_method: int, default ``cv2.CHAIN_APPROX_NONE``
        Contour method for contouring operation.
        
    Returns
    -------
    tuple:
        A tuple (``numpy.ndarray``, ``list``) that consists of image and list.
        
    Examples
    --------
    >>> from lfdtrack import *
    >>> img = cv2.imread('~/path/to/image.png')
    >>> I = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    >>> weldment, contour = weldment_roi(I)

    """
    
    # Converting image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Using median blur to blur the image
    img_blur = cv2.medianBlur(img_gray, blur_kernel_size)
    
    # Binarization using adaptive threshold
    img_thresh = cv2.adaptiveThreshold(img_blur,
                                       maxValue=binarization_max_value,
                                       adaptiveMethod=binarization_adaptive_method,
                                       thresholdType=binarization_threshold_type,
                                       blockSize=binarization_block_size,
                                       C=binarization_constant
                                      )
    
    # Dilating the image
    dilation_kernel = np.ones(dilation_kernel_size, np.uint8)
    
    img_dil = cv2.dilate(img_thresh, 
                         kernel=dilation_kernel, 
                         iterations=dilation_iterations
                        )
    # Eroding the image to remove blobs
    erosion_kernel = np.ones(erosion_kernel_size, np.uint8)
    
    img_erode = cv2.erode(img_dil,
                          kernel=erosion_kernel,
                          borderType=erosion_border_type,
                          iterations=erosion_iterations
                         )
    
    # Dilating again to connect borders
    img_dil2 = cv2.dilate(img_erode, 
                          kernel=dilation_kernel, 
                          iterations=dilation_iterations
                        )
    
    # Contour method
    contours, hierarchy = cv2.findContours(image=img_dil2,
                                           mode=contour_mode,
                                           method=contour_method
                                          )
    # Finding the long contour
    contour_length = []

    for i in contours:
        contour_length.append(len(i))

    long_contour_index = contour_length.index(max(contour_length))
    
    long_contour = contours[long_contour_index]
    
    # Polyfill image
    img_zero = np.zeros_like(image)
    
    img_poly = cv2.fillPoly(img_zero,
                            pts=[long_contour],
                            color=(255,255,255)
                           )
    
    weldment_roi = cv2.bitwise_and(img_poly, image)
    
    return (weldment_roi, long_contour)

def weldment_fingertrack(contours, fingertips):
    """Tracking fingers on the weldments.
    
    Parameters
    ----------
    contours: list
        A list of all the contour points.
    fingetips: dict
        A dictionary of fingertips co-ordinates.
        
    Returns
    -------
    dict:
        A dictionary of ``x`` and ``y`` co-ordinates of the fingertip over the weldment object.
    
    """
    
    con_x = []
    con_y = []
    weld_fingertips = {'x': [], 'y': []}
    
    for contour in contours:
        con_y.append(contour[0][1])
        con_x.append(contour[0][0])
        
    for x, y in zip(fingertips['x'], fingertips['y']):
        if (x < max(con_x) and x > min(con_x)) and (y < max(con_y) and y > min(con_y)):
            weld_fingertips['x'].append(x)
            weld_fingertips['y'].append(y)
            
    return weld_fingertips