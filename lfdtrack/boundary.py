import numpy as np
import sys


def upper_left_value(I):
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
    >>> I = np.array([[1,0,1], [1,0,1]])
    >>> I = np.pad(I, 1)
    >>> upper_left_value(I)
    (1, 1)
    
    """
    for row in range(0, I.shape[0]):
        for col in range(0, I.shape[1]):
            if I[row][col] == 1:
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
    >>> I = np.array([[1,1,0], [1,0,1], [0,1,0]])
    >>> I = np.pad(I, 1)
    >>> neighbors_8(I, (1,1), (0,0))
    [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
    
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

def find_first_value(I, neighbors):
    """Returns the positional co-ordinates
    of new found value point (b) and the new
    position of background point (c).
    
    The function is a part of the Moore boundary
    tracing algorithm.
    
    Parameters
    ----------
    I: np.ndarray
        A numpy array.
        This array represents the binary image.        
    neighbors: list
        A list of all the 8 neighbors of the central
        pixel starting (b) from the background point (c).
    value: int, default ``1``
        The value to search.
    
    Returns
    -------
    tuple
        Co-ordinates of new central point and the background point.
    
    Examples
    --------
    >>> from lfdtrack import *
    >>> I = np.array([[1,1,0], [1,0,1], [0,1,0]])
    >>> I = np.pad(I, 1)
    >>> n = neighbors_8(I, (1,1), (0,0))
    >>> find_first_value(I, n)
    (3, 2)

    """
    
    for n in enumerate(neighbors):
        i, j = n[1]
        # Checking if the co-ordinate matches 1
        if I[i][j] == 1:
            nk = n[0]
            nk_1 = n[0] - 1
            break
    try:        
        b = nk
        c = nk_1   
        return b, c
    except Exception as e:
        print(e)
        return None
        

def boundary_tracer(I):
    """Returns the co-ordinates of the boundary pixels.
    
    This algorithm is based on Moore Boundary Tracing Algorithm.
    
    Parameters
    ----------
    I: np.ndarray
        A numpy array.
        This array represents the binary image.
    
    Returns
    -------
    set
        A set of all the boundary co-ordinates.

    Examples
    --------
    >>> from lfdtrack import *
    >>> I = np.array([[1,1,0], [1,0,1], [0,1,0]])
    >>> boundary_tracer(I)
    {(1, 1), (1, 2), (2, 1), (2, 3), (3, 2)}

    """
    # Add Padding to the image 
    I = np.pad(I, 1)

    # Step 1: Starting with the uppermost-leftmost point
    b0 = upper_left_value(I)
    
    # Assigning the background point 
    c0 = (b0[0], b0[1]-1)
    
    # Step 2: Assigning the first points to boundary and background point
    b = b0
    c = c0
    
    # A set of boundary co-ordinates
    pad_boundary = set()
    
    # Keep running the till the last boundary co-ordinate detected matches the first
    while True:
        # Step 3: Finding neighboring co-ordinate to boundary pixel.
        n = neighbors_8(I, b, c)
        
        # Step 4: Assigning new b and c values
        try:
            nk, nk_1 = find_first_value(I, n)
        except Exception as e:
            print(e)
            sys.exit(1)

        b = n[nk]
        c = n[nk_1]
        
        # Stop the loop when the first boundary co-ordinate is detected
        if b in pad_boundary:
            break
        else:
            pad_boundary.add(b)

    # Creating a new un padded boundary
    boundary = set()
    
    # Subtracting one (-1) from the co-ordinate to compensate for
    # adding one layer of padding.
    for c in list(pad_boundary):
        unpadded_coord = (c[0]-1, c[1]-1)
        boundary.add(unpadded_coord)
    
    return boundary

def isolate_point_roi(I, boxes):
    """Returns a list of all the roi cropped images around the fingertip.
    
    The cropped images use the point box to isolate the region.
    
    Parameters
    ----------
    I: np.ndarray
        A numpy array.
        This array represents the binary image.
    
    boxes: list
        A list of all the box co-ordinates in the format used
        for polyfill function.
        **Example**: ``boxes = [np..array([[483,327], [483, 527], [683, 527], [683, 327]])]``
        
        The co-ordinates specified are as ``[[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]``.
    
    Returns
    -------
    list
        A list of cropped roi images from the input image.
        This list consist of numpy array.
    
    Examples
    --------
    >>> from lfdtrack import *
    >>> I = np.zeros([5,5])
    >>> boxes = [np.array([[[1,1],[1,3],[3,3],[3,1]]])]
    >>> roi_imgs = isolate_point_roi(I, boxes)
    
    """
    roi_images = []

    for box in boxes:
        # Columns
        xmin = box[0][0][0]
        xmax = box[0][2][0]
        
        # Rows
        ymin = box[0][0][1]
        ymax = box[0][2][1]

        roi_images.append(I[ymin:ymax,xmin:xmax])
        
    return roi_images