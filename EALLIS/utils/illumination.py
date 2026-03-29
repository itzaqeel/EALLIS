import numpy as np


def brightness(img):
    return np.mean(img)


def classify_light(b):
    if b < 40:
        return "extreme"
    elif b < 80:
        return "dark"
    else:
        return "normal"
