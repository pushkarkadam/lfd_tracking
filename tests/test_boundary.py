import pytest 
import sys 
import numpy as np

sys.path.append('../')

from lfdtrack import * 


def test_upper_left_value():
    """Test for upper_left_value()"""
    I = np.array([[1,0,1], [1,0,1]])
    I = np.pad(I, 1)
    coord = upper_left_value(I)

    assert(coord == (1,1))

def test_neighbors_8():
    """Test for neighbors_8()"""
    I = np.array([[1,1,0], [1,0,1], [0,1,0]])
    I = np.pad(I, 1)
    neighbor = neighbors_8(I, (1,1), (0,0))
    
    assert(neighbor == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)])

def test_find_first_value():
    """Test for find_first_value()"""
    I = np.array([[1,1,0], [1,0,1], [0,1,0]])
    I = np.pad(I, 1)
    n = neighbors_8(I, (1,1), (0,0))
    coord = find_first_value(I, n)

    assert(coord == (3, 2))

def test_boundary_tracer():
    """Test for boundary tracer"""
    I = np.array([[1,1,0], [1,0,1], [0,1,0]])
    boundary = boundary_tracer(I)

    assert(boundary == {(0,0), (0,1), (1,2), (2,1), (1,0)})