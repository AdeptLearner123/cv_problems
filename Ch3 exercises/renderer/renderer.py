import numpy as np

from .transform import rotate, project, to_pixel
from .rasterize import rasterize
from .pyramid import get_pyramid

def render(
            points_list, triangles_list, uvs_list, img,
            angleY, angleX, angleZ,
            x0, x1, y0, y1, w, h):
    points = np.transpose(np.vstack(points_list))
    triangles = np.vstack(triangles_list)
    uvs = np.vstack(uvs_list)

    points = rotate(points, angleY, angleX, angleZ)
    screen_pts = project(points)
    pixel_pts = to_pixel(screen_pts, x0, x1, y0, y1, w, h)

    layers, _ = get_pyramid(img, 2)
    return rasterize(pixel_pts, triangles, uvs, layers, w, h)
