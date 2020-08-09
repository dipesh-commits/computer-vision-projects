import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('5.jpg')

width = 640
height = 480

img = cv2.resize(img,(640,480))
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray,(5,5),1)

img_canny = cv2.Canny(img_blur,40,60)

kernel = np.ones((5,5))

img_dilate = cv2.dilate(img_canny,kernel,iterations=2)

img_erosion = cv2.erode(img_dilate, kernel, iterations=1)

img2 = img.copy()
img3 = img.copy()


contours, hierarchy = cv2.findContours(img_erosion,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


biggest = np.array([])
max_area = 0
for contour in contours:
	area = cv2.contourArea(contour)
	if area>5000:
		perimeter = cv2.arcLength(contour,True)
		approx = cv2.approxPolyDP(contour,0.02*perimeter,True)
		if area>max_area:
			biggest = approx
			max_area = area


biggest = biggest.reshape((len(biggest),2))

def swaplist(mylist,pos1,pos2):
	print(mylist[pos1],mylist[pos2])
	mylist = list(mylist)
	temp = mylist.pop(pos1)
	temp1 = mylist.pop(pos2-1)
	mylist.insert(pos1,temp1)		
	mylist.insert(pos2,temp)
	return mylist



cv2.drawContours(img3,contours,-1,(0,255,0),3)

biggestNew = swaplist(biggest,0,1)
pt1 = np.float32(biggestNew)
pt2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

matrix = cv2.getPerspectiveTransform(pt1,pt2)
imgWarp = cv2.warpPerspective(img3,matrix,(width,height))


all_images = np.hstack([img3, imgWarp])
cv2.imshow('img',all_images)
cv2.waitKey(0)
cv2.destroyAllWindows()


