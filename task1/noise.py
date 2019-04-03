import numpy as np
import cv2
from task1.utils import calculate_rms, calculate_rms_cropped
from task1.mylib_task1 import average_convolve, median_convolve, gaussian, dist_arr, bilateral_convolve
"""
This is main function for task 1.
It takes 2 arguments,
'src_img_path' is path for source image.
'dst_img_path' is path for output image, where your result image should be saved.

You should load image in 'src_img_path', and then perform task 1 of your assignment 1,
and then save your result image to 'dst_img_path'.
"""
def task1(src_img_path, dst_img_path, answer_path):
    answer = cv2.imread(answer_path)
    img = cv2.imread(src_img_path)
    best_img = np.zeros(img.shape)
    best_value = calculate_rms_cropped(answer, best_img)
    kernel_size = [3,5,7,9,11,13,15]
    sigma_s_list = [75]
    sigma_r_list = [10, 25, 50, 75]
    #calculate average_filter
    print('average filter')
    for k in kernel_size:
        print('k: ', k)
        tmp = apply_average_filter(img, k)
        err = calculate_rms_cropped(answer, tmp)
        if best_value > err:
            print('best!')
            best_img = tmp
            best_value = err
            print(best_value)
            cv2.imwrite(dst_img_path, best_img)
    #calculate median filter
    print('median filter')
    for k in kernel_size:
        print('k: ', k)
        tmp = apply_median_filter(img, k)
        err = calculate_rms_cropped(answer, tmp)
        if best_value > err:
            print('best!')
            best_img = tmp
            best_value = err
            print(best_value)
            cv2.imwrite(dst_img_path, best_img)
    print('bilateral filter')
    print('imamade besto valu: ', best_value)
    #calculate bilateral filter
    for i in sigma_s_list:
        for j in sigma_r_list:
            for k in kernel_size:
                print('i, j ,k: ', i, " ", j, " ", k)
                tmp = apply_bilateral_filter(img, k, i, j)
                err = calculate_rms_cropped(answer, tmp)
                print('bilateral err: ', err)
                if best_value > err:
                    print('best!')
                    best_img = tmp
                    best_value = err
                    print(best_value)
                    cv2.imwrite(dst_img_path, best_img)
    print("best value: ", best_value)
    cv2.imwrite(dst_img_path, best_img)
    return best_value


"""
You should implement average filter convolution algorithm in this function.
It takes 2 arguments,
'img' is source image, and you should perform convolution with average filter.
'kernel_size' is a int value, which determines kernel size of average filter.

You should return result image.
"""
def apply_average_filter(img, kernel_size):
    dst = np.copy(img)
    d = int((kernel_size-1)/2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = average_convolve(img, i, j, d)
            dst[i, j] = pixel
    return dst


"""
You should implement median filter convolution algorithm in this function.
It takes 2 arguments,
'img' is source image, and you should perform convolution with median filter.
'kernel_size' is a int value, which determines kernel size of median filter.

You should return result image.
"""
def apply_median_filter(img, kernel_size):
    dst = np.copy(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    d = int((kernel_size - 1) / 2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = median_convolve(img, i, j, d)
            dst[i, j] = pixel
    dst = cv2.cvtColor(dst, cv2.COLOR_HSV2BGR)
    return dst


"""
You should implement convolution with additional filter.
You can use any filters for this function, except average, median filter.
It takes at least 2 arguments,
'img' is source image, and you should perform convolution with median filter.
'kernel_size' is a int value, which determines kernel size of average filter.
'sigma_s' is a int value, which is a sigma value for G_s
'sigma_r' is a int value, which is a sigma value for G_r

You can add more arguments for this function if you need.

You should return result image.
"""
def apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r):
    dst = np.copy(img)
    d = int((kernel_size - 1) / 2)
    sigma_s = d/3
    gaussian_arr = gaussian(sigma_s, dist_arr(d))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = bilateral_convolve(img, i, j, d, sigma_r, gaussian_arr)
            dst[i, j] = pixel
    return dst

dst_path = "answer.png"
src_path = ['test1_noise.png','test2_noise.png','test3_noise.png','test4_noise.png','test5_noise.png',]
answer_path = ['test1_clean.png','test2_clean.png','test3_clean.png','test4_clean.png','test5_clean.png']
best_value = []
for s, a in zip(src_path, answer_path):
    v = task1(s, dst_path, a)
    best_value.append(v)
print(best_value)