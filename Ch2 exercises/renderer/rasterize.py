import numpy as np


def rasterize(pixel_pts, triangles, w, h):
    canvas = np.zeros((h, w))

    for i in range(triangles.shape[0]):
        i0, i1, i2 = triangles[i, :]
        draw_triangle(
            canvas,
            pixel_pts[:2, i0],
            pixel_pts[:2, i1],
            pixel_pts[:2, i2])
    return canvas


def draw_triangle(canvas, p0, p1, p2):
    # Point supplied counter-clock-wise
    # Points are in 2D pixel space, no depth

    h, w = canvas.shape

    y_vals = np.arange(0, h)
    x_vals = np.arange(0, w)

    xv, yv = np.meshgrid(x_vals, y_vals)
    edge01 = edge_func(xv, yv, p0, p1)
    edge12 = edge_func(xv, yv, p1, p2)
    edge20 = edge_func(xv, yv, p2, p0)

    canvas[(edge01 <= 0) & (edge12 <= 0) & (edge20 <= 0)] = 1.


def edge_func(x, y, p0, p1):
    return (x - p0[0]) * (p1[1] - p0[1]) - (y - p0[1]) * (p1[0] - p0[0])