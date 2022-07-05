import numpy as np


def interpolate(p0, p1, t):
    return (1 - t) * p0 + t * p1


def interpolate2d(p00, p10, p01, p11, x, y):
    print("First")
    p0 = interpolate(p00, p10, x)
    print("second")
    p1 = interpolate(p01, p11, x)
    print("Third")
    return interpolate(p0, p1, y)


def interpolate2d_img(img, x, y):
    floor_y = np.floor(y).astype(np.int)
    floor_x = np.floor(x).astype(np.int)
    p00 = img[floor_y, floor_x, :]
    p10 = img[floor_y, floor_x + 1, :]
    p01 = img[floor_y + 1, floor_x, :]
    p11 = img[floor_y + 1, floor_x + 1, :]
    x_interp = x - floor_x
    y_interp = y - floor_y
    return interpolate2d(p00, p10, p01, p11, x_interp[:, np.newaxis], y_interp[:, np.newaxis])


def interpolate3d_img(img0, img1, x, y, mips):
    interp0 = interpolate2d_img(img0, x, y)
    interp1 = interpolate2d_img(img1, x, y)
    return interpolate(interp0, interp1, mips[:, np.newaxis])