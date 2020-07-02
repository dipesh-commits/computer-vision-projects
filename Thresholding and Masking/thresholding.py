import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('swan.jpg')
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#binary thresholding
ret0, threshold = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)

#binary inverse thresholding
ret1, inverse_threshold = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY_INV)

#adpative thresholding
adaptive_threshold = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,129,5)

#otsu binarization
ret2,otsu = cv2.threshold(gray_img,0,255,cv2.THRESH_OTSU)

cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 2048,1024)
all_image = np.hstack((gray_img,threshold,inverse_threshold,adaptive_threshold,otsu))
cv2.imshow('Image',all_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


