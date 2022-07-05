import numpy as np

from .transform import rotate, project, to_pixel
from .rasterize import rasterize

def render(
            points_list, triangles_list,
            angleY, angleX, angleZ,
            x0, x1, y0, y1, w, h):
    points = np.transpose(np.vstack(points_list))
    triangles = np.vstack(triangles_list)

    points = rotate(points, angleY, angleX, angleZ)
    screen_pts = project(points)
    pixel_pts = to_pixel(screen_pts, x0, x1, y0, y1, w, h)

    canvas = rasterize(pixel_pts, triangles, w, h)
    canvas = np.flipud(canvas) # lower Y should be on bottom of image
    return canvas