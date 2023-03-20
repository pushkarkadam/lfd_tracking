import numpy as np
import matplotlib.pyplot as plt
import PIL

def point_plotter():
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True


    points = []

    def mouse_event(event):
        print('x: {} and y: {}'.format(event.xdata, event.ydata))
        points.append((event.xdata, event.ydata))


    im = plt.imread("../data/images/paint_weld.png")

    print(im.shape)

    fig, ax = plt.subplots()

    im = ax.imshow(im, extent=[0, im.shape[1],im.shape[0],0])

    cid = fig.canvas.mpl_connect('button_press_event', mouse_event)

    plt.show()

    return points