import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_point_clicker import clicker


def point_plotter(image_path, filename=''):
    """Returns a list of point co-ordinates plotted on the image.

    Opens up a GUI to plot the points.

    Parameters
    ----------
    image_path: str
        Path to the image.

    filename: str, default ``''``
        Path to store the points data.
        Use the ``.pkl`` extension.
        Usage: ``filename="~/path/to/filename.pkl"``

    Returns
    -------
    list 
        A list of tuples with x and y co-ordinates on the image.

    """
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True

    # storing co-ordinates
    points = []

    def mouse_event(event):
        print('x: {} and y: {}'.format(event.xdata, event.ydata))
        points.append((event.xdata, event.ydata))

    # Reading image path
    im = plt.imread(image_path)

    fig, ax = plt.subplots()

    im = ax.imshow(im, extent=[0, im.shape[1],im.shape[0],0])

    cid = fig.canvas.mpl_connect('button_press_event', mouse_event)

    plt.show()

    if filename:
        with open(filename, 'wb') as f:
            pickle.dump(points, f)

    return points

def path_planner(image_path, filename=''):
    """Returns a dictionary of x and y co-ordinates.

    Parameters
    ----------
    image_path: str
        Path to the image.

    filename: str, default ``''``
        Path to store the points data.
        Use the ``.pkl`` extension.
        Usage: ``filename="~/path/to/filename.pkl"``

    Returns
    -------
    dict
        A dictionary of x and y values of the points.

    """

    fig, ax = plt.subplots(constrained_layout=True)

    # Reading image from the path
    image = plt.imread(image_path)

    ax.imshow(image, cmap="gray")
    klicker = clicker(
    ax,
    ["points"],
    markers=["o"],
    linestyle= "--",
    colors=['r']
    )

    plt.show()

    coords = klicker.get_positions()

    points = {'x': [], 'y': []}

    for coord in list(coords['points']):
        points['x'].append(np.array(coord)[0])
        points['y'].append(np.array(coord)[1])

    if filename:
        with open(filename, 'wb') as f:
            pickle.dump(points, f)
        
    return points
