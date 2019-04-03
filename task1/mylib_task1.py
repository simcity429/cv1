import cv2
import numpy as np
from math import sqrt, pi, exp, pow

def calculate_rms(img1, img2):
    if (img1.shape[0] != img2.shape[0]) or \
            (img1.shape[1] != img2.shape[1]) or \
            (img1.shape[2] != img2.shape[2]):
        raise Exception("img1 and img2 should have sime sizes.")

    diff = np.abs(img1 - img2)
    return np.sqrt(np.mean(diff ** 2))


def dist_arr(d):
    ret = np.zeros((2*d+1, 2*d+1))
    for i in range(-d, d+1):
        for j in range(-d, d+1):
            ret[i+d, j+d] = sqrt(i**2 + j**2)
    return ret

def color(img_patch, center, d):
    ret = np.zeros((2*d + 1, 2*d + 1, 3))
    ret[:, :] = center
    ret = np.abs(ret - img_patch)
    return ret

def gaussian(sigma, x):
    return np.exp(-x**2/(2*(sigma**2)))

def get_patch(img, x, y, d):
    flag = False
    x_size = img.shape[0]
    y_size = img.shape[1]
    if x-d < 0 or y-d < 0 or x+d >= x_size or y+d >= y_size:
        flag = True
    if flag:
        ret = np.zeros((2*d+1, 2*d+1, 3))
        for i in range(x - d, x + d + 1):
            for j in range(y - d, y + d + 1):
                ret[i-(x-d), j-(y-d)] = get_pixel(img, i, j)
        return ret
    else:
        return img[x-d:x+d+1, y-d:y+d+1]


def get_pixel(img, x, y):
    x_size = img.shape[0]
    y_size = img.shape[1]
    if x >= x_size:
        x = 2*x_size - x - 2
    if y >= y_size:
        y = 2*y_size - y - 2
    return img[abs(x), abs(y)]

def average_convolve(img, x, y, d):
    img_patch = get_patch(img, x, y, d)
    return np.mean(img_patch, axis=(0,1))

def median_convolve(img, x, y, d):
    return np.median(get_patch(img, x, y, d).reshape((-1, 3)), axis=0)


def bilateral_convolve(img, x, y, d, sigma_r, gaussian_arr):
    center = get_pixel(img, x, y)
    img_patch = get_patch(img, x, y, d)
    conv_arr = gaussian_arr.reshape((2*d+1, 2*d+1, 1)) * gaussian(sigma_r, color(img_patch, center, d))
    conv_arr /= np.sum(conv_arr, axis=(0,1))
    return np.sum(conv_arr*img_patch, axis=(0,1))

def apply_average_filter(img, kernel_size):
    dst = np.copy(img)
    d = int((kernel_size-1)/2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = average_convolve(img, i, j, d)
            dst[i, j] = pixel
    return dst

def apply_median_filter(img, kernel_size):
    dst = np.copy(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    d = int((kernel_size-1)/2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = median_convolve(img, i, j, d)
            dst[i, j] = pixel
    dst = cv2.cvtColor(dst, cv2.COLOR_HSV2BGR)
    return dst

def apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r):
    dst = np.copy(img)
    d = int((kernel_size-1)/2)
    sigma_s = int(d/3)
    gaussian_arr = gaussian(sigma_s, dist_arr(d))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = bilateral_convolve(img, i, j, d, sigma_r, gaussian_arr)
            dst[i, j] = pixel
    return dst


