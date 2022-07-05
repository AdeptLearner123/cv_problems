import numpy as np
from scipy.signal import convolve2d


def upsample(img):
    h, w, c = img.shape
    new_img = np.zeros((h * 2, w * 2, c))
    new_img[::2, ::2, :] = img
    new_img[1::2, ::2, :] = img
    new_img[::2, 1::2, :] = img
    new_img[1::2, 1::2, :] = img
    return new_img


def get_pyramid(img, levels):
    l = np.array([1, 4, 6, 4, 1]) / 16.
    kernel = np.transpose([l]) @ [l]

    layers = [img]
    band_passes = []
    for i in range(levels):
        decimated = np.zeros(layers[-1].shape)
        decimated[:, :, 0] = convolve2d(layers[-1][:, :, 0], kernel, 'same', 'symm')
        decimated[:, :, 1] = convolve2d(layers[-1][:, :, 1], kernel, 'same', 'symm')
        decimated[:, :, 2] = convolve2d(layers[-1][:, :, 2], kernel, 'same', 'symm')

        band_pass = layers[-1] - decimated

        layers.append(decimated.astype(np.uint8))
        band_passes.append(band_pass.astype(np.uint8))
    return np.array(layers), np.array(band_passes)