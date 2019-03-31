import numpy as np

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