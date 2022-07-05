import numpy as np
import math


def rotate(points, angleY, angleX, angleZ):
    # Rotate order is Y -> X -> Z
    # Z points into screen, X to the right, Y upwards
    # Rotation center is (0, 0, 1)

    rotZ = np.array([
        [math.cos(angleZ),  -math.sin(angleZ),  0],
        [math.sin(angleZ),  math.cos(angleZ),   0],
        [0,                 0,                  1]
    ])

    rotX = np.array([
        [1, 0,                  0],
        [0, math.cos(angleX),   math.sin(angleX)],
        [0, -math.sin(angleX),  math.cos(angleX)]
    ])

    rotY = np.array([
        [math.cos(angleY),  0,  -math.sin(angleY)],
        [0,                 1,  0],
        [math.sin(angleY),  0,  math.cos(angleY)]
    ])

    center = np.array([
        [0],
        [0],
        [1]
    ])

    return rotY @ rotX @ rotZ @ (points - center) + center


def project(points):
    # Project into screen space

    screen_pts = np.zeros(points.shape)
    z_row = points[2, :]
    screen_pts[:2, :] = points[:2, :] / z_row
    screen_pts[2, :] = 1 / z_row
    return screen_pts


def to_pixel(screen_pts, x0, x1, y0, y1, w, h):
    # From screen space to pixel space

    pixel_pts = np.zeros(screen_pts.shape)
    pixel_pts[0, :] = (screen_pts[0, :] - x0) / (x1 - x0) * w
    pixel_pts[1, :] = (screen_pts[1, :] - y0) / (y1 - y0) * h
    pixel_pts[2, :] = screen_pts[2, :]
    return pixel_pts