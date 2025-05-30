import gc
import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def fig_to_numpy(fig):

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    image = Image.open(buf)
    return np.array(image)


def matplotlib_clear(fig):
    # https://stackoverflow.com/questions/31156578/
    # matplotlib-doesnt-release-memory-after-savefig-and-close

    # Clear the current axes.
    plt.cla()
    # Clear the current figure.
    plt.clf()
    # Closes all the figure windows.
    plt.close("all")
    plt.close(fig)
    gc.collect()


def pixels_to_inches(pixels, ppi=91.79):
    """
    caculate inches based on the following formula
    inches = pixels / ppi

    How to estimate ppi?
        given a 24" FHD monitor
        diagonal pixels = (1920 * 1920 + 1080 * 1080) ^ 0.5
        ppi = diagonal pixels / 24 (approximate 91.79)
    """

    if not np.iterable(pixels):
        pixels = [pixels]

    for pixel in pixels:
        if pixel < 0:
            raise ValueError

    if ppi < 0:
        raise ValueError

    return [1.0 * pixel / ppi for pixel in pixels]
