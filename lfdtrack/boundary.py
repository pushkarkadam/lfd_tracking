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

def neighbors_8(I, pixel_coords, background_point):
    """Returns a dictionary of pixel co-ordinates and their respective values.
    
    Orders the neighbors in sequence.
    The sequence starts with the background point.
    The algorithm is a part of Moore boundary tracing algorithm.
    Where the n1 must start from C (``backgorund_point``).
    The algorithm uses clockwise checking.
    
    Parameters
    ----------
    I: np.ndarray
        A numpy array.
        This array represents the binary image.
    pixel_coords: tuple
        The central pixel co-ordinates.
        Represented as b in the algorithm.
    background_point: tuple
        The co-ordinate of the point that traces the boundary.
        Represented as c in the algorithm
        
    Returns
    -------
    list
        A list of all the 8 neighbor co-ordinates in a sequence followed
        by the background point.
        
    Examples
    --------
    >>> from lfdtrack import *
    >>> I = np.array([[1,2,3], [4,4,5]])
    >>> I = np.pad(I, 1)
    >>> neighbors_8(bar, (1,1), (1,2))
    [(1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0), (0, 1), (0, 2)]
    
    """
    
    i, j = pixel_coords
    c = background_point
    
    n1 = (i-1, j-1)
    n2 = (i-1, j)
    n3 = (i-1, j+1)
    n4 = (i, j+1)
    n5 = (i+1, j+1)
    n6 = (i+1, j)
    n7 = (i+1, j-1)
    n8 = (i, j-1)
    
    neighbors = [n1, n2, n3, n4, n5, n6, n7, n8]
    
    # Finding the location of the C element
    c_loc = neighbors.index(c)
    
    # slicing the list
    after = neighbors[:c_loc]
    before = neighbors[c_loc:]
    
    # Matching the list
    sequence = before + after
    
    return sequence