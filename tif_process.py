import cv2 as cv
import numpy as np


path = r'/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/image/GF2_PMS1_E104.0_N1.3_20240422_L1A13799631001-MSS1.tiff'

def print_result(arr, name):
    
    ch1 = arr[:, :, 0]
    ch2 = arr[:, :, 1]
    ch3 = arr[:, :, 2]
    ch4 = arr[:, :, 3]
    print(ch1.shape)
    print('{} unqiue 1:'.format(name), np.unique(ch1)[:10])
    print('{} unqiue 2:'.format(name), np.unique(ch2)[:10])
    print('{} unqiue 3:'.format(name), np.unique(ch3)[:10])
    print('{} unqiue 4:'.format(name), np.unique(ch4)[:10])


def cv_load():
    arr = cv.imread(path, cv.IMREAD_UNCHANGED)
    print_result(arr, 'cv')
    print(arr.shape)

cv_load()

