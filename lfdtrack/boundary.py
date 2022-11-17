import numpy as np


def upper_left_value(I, value=1):
    """Returns the coordinate of the uppermost-leftmost pixel value given as input.
    
    Iterates over the matrix from top to bottom
    and left to right.
    The process stops when the first value is detected.
    
    Parameters
    ----------
    I: np.ndarray
        A numpy array.
        This array represents the binary image 
    value: int, default ``1``,
        The value to detect.
        This value is default to ``1``. 
        But the user still has the option to use some other number.
        
    Returns
    -------
    b0: tuple
        A tuple indicating the position of the uppermost-leftmost pixel.
        
    Examples
    --------
    >>> from lfdtrack import *
    >>> I = np.array([[1,2,3], [4,4,5]])
    >>> upper_left_value(I, 1)
    (0,0)
    
    """
    for row in range(0, I.shape[0]):
        for col in range(0, I.shape[1]):
            if I[row][col] == value:
                b0 = (row, col)
                return b0