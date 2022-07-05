import numpy as np
import numpy.linalg
import cv2
import math
import colorsys


def edge_func(x, y, p0, p1):
    return (x - p0[0]) * (p1[1] - p0[1]) - (y - p0[1]) * (p1[0] - p0[0])


def get_mask(h, w, p0, p1, p2, p3):
    x = np.arange(w)
    y = np.arange(h)
    xv, yv = np.meshgrid(x, y)
    e0 = edge_func(xv, yv, p0, p1)
    e1 = edge_func(xv, yv, p1, p2)
    e2 = edge_func(xv, yv, p2, p3)
    e3 = edge_func(xv, yv, p3, p0)

    return (e0 >= 0) & (e1 >= 0) & (e2 >= 0) & (e3 >= 0)


def get_homogenous_line(p0, p1):
    n = np.array([p1[1] - p0[1], p0[0] - p1[0]], dtype=float)
    n /= np.linalg.norm(n, 2)
    d = -n[0] * p0[0] - n[1] * p0[1]

    return np.array([n[0], n[1], d])


def get_diff_img(img):
    background = np.array([62, 71, 79]) / 255.
    diff_img = img - background
    diff_img[diff_img < 0] = 0
    return diff_img.astype(np.float32)


def get_hues(img, mask, line):
    x_vals, y_vals, distances = get_mask_points(img, mask, line)

    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    hues = hsv_img[y_vals, x_vals, 0]

    d_min = math.floor(distances.min())
    d_max = math.floor(distances.max())
    wheel = np.zeros((d_max - d_min + 1, 360))
    wheel[np.floor(distances).astype(np.int) - d_min, np.floor(hues).astype(np.int)] += 1

    return np.stack((distances, hues)), wheel, d_min, d_max


def get_avg_hues(wheel):
    indices = np.indices(wheel.shape)

    sins = np.sin(indices[1] / 180 * np.pi)
    coss = np.cos(indices[1] / 180 * np.pi)
    avg_sins = np.sum(sins * wheel, axis=1) / np.sum(wheel, axis=1)
    avg_coss = np.sum(coss * wheel, axis=1) / np.sum(wheel, axis=1)

    hues = np.arctan2(avg_sins, avg_coss) * 180 / np.pi
    hues[hues < 0] += 360
    return hues, avg_sins, avg_coss


def get_mask_points(img, mask, line):
    indices = np.indices(img.shape[:2])
    x_vals = indices[1][mask]
    y_vals = indices[0][mask]
    z_vals = np.ones(len(x_vals))
    points = np.stack((x_vals, y_vals, z_vals))

    distances = line[np.newaxis, :] @ points
    distances = distances[0]

    return x_vals, y_vals, distances


def get_rainbow_img(img, mask, line, d_max, rainbow_func):
    hsv_img = np.zeros(img.shape)
    x_vals, y_vals, distances = get_mask_points(img, mask, line)
    hues = rainbow_func(distances, d_max)

    hsv_img[y_vals, x_vals, 0] = hues
    hsv_img[y_vals, x_vals, 1] = 1
    hsv_img[y_vals, x_vals, 2] = 1
    hsv_img = hsv_img.astype(np.float32)

    bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return img


def get_d_max(line, p1):
    return math.floor(np.dot(line, [p1[0], p1[1], 1]))


def rainbow_linear(d, d_max):
    hues = 250 - 360 * d / d_max
    hues[hues < 0] += 360
    return hues
