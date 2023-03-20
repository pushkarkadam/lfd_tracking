import numpy as np
import matplotlib.pyplot as plt
import pickle


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