import numpy as np
import cv2

def denoise2(img):
    W = img.shape[0]
    H = img.shape[1]
    img = np.fft.fft2(img)
    f = generate_band_filter(50, 40, W, H, band_pass=False)
    img = img*f
    img = np.abs(np.fft.ifft2(img))
    return img

def circle(a, r, x, y, w, h, in_flag):
    for i in range(w):
        for j in range(h):
            if in_flag:
                if (i-x)**2 + (j-y)**2 <= r**2:
                    a[i,j] = 1
            else:
                if (i-x)**2 + (j-y)**2 <= r**2:
                    a[i, j] = 0
    return a

def generate_filter(threshold, W, H, low):
    ret = np.zeros((W, H))
    if low:
        ret = circle(ret, threshold, 0, 0, W, H, True)
        ret = circle(ret, threshold, 0, H-1, W, H, True)
        ret = circle(ret, threshold, W-1, 0, W, H, True)
        ret = circle(ret, threshold, W-1, H-1, W, H, True)
        return ret
    if not low:
        ret.fill(1)
        ret = circle(ret, threshold, 0, 0, W, H, False)
        ret = circle(ret, threshold, 0, H-1, W, H, False)
        ret = circle(ret, threshold, W-1, 0, W, H, False)
        ret = circle(ret, threshold, W-1, H-1, W, H, False)
        return ret

def generate_band_filter(threshold_1, threshold_2, W, H, band_pass):
    #threshold_1 > threshold_2
    ret = np.zeros((W, H))
    ret = circle(ret, threshold_1, 0, 0, W, H, True)
    ret = circle(ret, threshold_1, 0, H - 1, W, H, True)
    ret = circle(ret, threshold_1, W - 1, 0, W, H, True)
    ret = circle(ret, threshold_1, W - 1, H - 1, W, H, True)
    ret = circle(ret, threshold_2, 0, 0, W, H, False)
    ret = circle(ret, threshold_2, 0, H - 1, W, H, False)
    ret = circle(ret, threshold_2, W - 1, 0, W, H, False)
    ret = circle(ret, threshold_2, W - 1, H - 1, W, H, False)
    if band_pass is False:
        ret = np.where(ret>0, 0, 1)
    return ret


def fftshift(img):
    w = int(img.shape[0]/2)
    h = int(img.shape[1]/2)
    ret = np.copy(img)
    ret[0:w, 0:h] = img[w:, h:]
    ret[w:, h:] = img[0:w, 0:h]
    ret[w:, 0:h] = img[0:w, h:]
    ret[0:w, h:] = img[w:, 0:h]
    return ret

def spectrum(img):
    return 10* fftshift(np.log(np.abs(img) + 1e-10))

sample_name = "task2_sample.png"
c_1 = cv2.imread("task2_corrupted_1.png", cv2.IMREAD_GRAYSCALE)
c_2 = cv2.imread("task2_corrupted_2.png", cv2.IMREAD_GRAYSCALE)
sample = cv2.imread(sample_name, cv2.IMREAD_GRAYSCALE)
W = sample.shape[0]
H = sample.shape[1]
img = np.fft.fft2(c_2)
low = generate_band_filter(42, 37, W, H, False)
img = img*low
cv2.imwrite("ah.jpg", spectrum(img))
img = np.abs(np.fft.ifft2(img))
cv2.imwrite("oh.png", img)
