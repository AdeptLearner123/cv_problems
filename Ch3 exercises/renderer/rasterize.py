import numpy as np
from .interpolate import interpolate2d, interpolate2d_img, interpolate3d_img


TESTX = 512
TESTY = 512

def rasterize(pixel_pts, triangles, pt_uvs, pyramid, w, h):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    mip_canvas = np.zeros((h, w))

    for i in range(triangles.shape[0]):
        i0, i1, i2 = triangles[i, :]
        uvs, duvs, mask = get_uvs(
            h,
            w,
            pixel_pts[:, i0],
            pixel_pts[:, i1],
            pixel_pts[:, i2],
            pt_uvs[i0, :],
            pt_uvs[i1, :],
            pt_uvs[i2, :]
        )
        mipmap = calculate_mipmap(duvs)
        draw_triangle(canvas, uvs, mipmap, mask, pyramid)
        print(duvs.max())
        draw_triangle_uvs(mip_canvas, mipmap, mask)

    canvas = np.flipud(canvas) # lower Y should be on bottom of image
    mip_canvas = np.flipud(mip_canvas)
    return canvas, mip_canvas


def draw_triangle(canvas, uvs, mipmap, mask, pyramid):
    print(mipmap.max(), mipmap.min())
    mask0 = mask & (mipmap >= 0) & (mipmap < 1)
    canvas[mask0, :] = interpolate3d_img(pyramid[0], pyramid[1], uvs[mask0, 1], uvs[mask0, 0], mipmap[mask0])
    mask1 = mask & (mipmap >= 1) & (mipmap < 2)
    canvas[mask1, :] = interpolate3d_img(pyramid[1], pyramid[2], uvs[mask1, 1], uvs[mask1, 0], mipmap[mask1] - 1)


"""
def draw_triangle(canvas, uvs, mipmap, mask, pyramid):
    print(mipmap.max(), mipmap.min())
    uvs /= mipmap[:, :, np.newaxis] + 1
    uvs = np.floor(uvs).astype(np.int)
    print(pyramid[mipmap[mask]][0].shape)
    print(pyramid[mipmap[mask]][1].shape)

    mask0 = mask & (mipmap == 0)
    canvas[mask0, :] = pyramid[0][uvs[mask0, 0], uvs[mask0, 1], :]

    mask1 = mask & (mipmap == 1)
    canvas[mask1, :] = pyramid[1][uvs[mask1, 0], uvs[mask1, 1], :]
"""


def draw_triangle_uvs(canvas_dx, duvs, mask):
    canvas_dx[mask] = duvs[mask]


def calculate_mipmap(duvs):
    p = duvs.max(axis = -1)
    p = np.log2(p)
    p[p < 0] = 0
    return p


def get_uvs(h, w, p0, p1, p2, uv0, uv1, uv2):
    y_vals = np.arange(0, h)
    x_vals = np.arange(0, w)

    xv, yv = np.meshgrid(x_vals, y_vals)

    e = np.zeros((h, w, 3))
    e[:, :, 0] = edge_func(xv, yv, p1, p2)
    e[:, :, 1] = edge_func(xv, yv, p2, p0)
    e[:, :, 2] = edge_func(xv, yv, p0, p1)

    # Perspective correction
    l = e * [p0[2], p1[2], p2[2]]
    total = np.sum(l, axis=-1)
    l /= total[:, :, np.newaxis]

    mask = np.all((e <= 0), axis=-1)
    uvs = np.zeros((h, w, 2))
    uv_arr = np.reshape([uv0, uv1, uv2], (1, 1, 3, 2))
    uvs[mask, :] = np.sum(l[mask, :, np.newaxis] * uv_arr, axis=2)

    # Derivatives
    a = np.zeros((h, w, 3, 2))
    a[:, :, 0] = edge_func_derivative(h, w, p1, p2)
    a[:, :, 1] = edge_func_derivative(h, w, p2, p0)
    a[:, :, 2] = edge_func_derivative(h, w, p0, p1)

    a *= np.reshape([p0[2], p1[2], p2[2]], (1, 1, 3, 1))
    a /= total[:, :, np.newaxis, np.newaxis]

    dl = np.zeros(a.shape)
    dtotal = np.sum(a, axis=2)
    dl = a - dtotal[:, :, np.newaxis, :] * l[:, :, :, np.newaxis]
    duvs = np.zeros((h, w, 2))
    uv_arr = uv_arr[:,:,:,::-1]
    duvs[mask] = np.sum(dl[mask] * uv_arr, axis=2)

    return uvs, duvs, mask


def edge_func(x, y, p0, p1):
    return (x - p0[0]) * (p1[1] - p0[1]) - (y - p0[1]) * (p1[0] - p0[0])


def edge_func_derivative(h, w, p0, p1):
    arr = np.zeros((h, w, 2))
    arr[:, :, 0] = p1[1] - p0[1]
    arr[:, :, 1] = p0[0] - p1[0]
    return arr