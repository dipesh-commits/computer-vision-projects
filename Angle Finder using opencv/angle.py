import cv2
import math
import numpy as np

img_path = 'test2.png'

img = cv2.imread(img_path)


points = []


def slope(p1,p2):
	slope = (p2[1]-p1[1])/(p2[0]-p1[0])
	return slope


def find_angle(points):
	p1,p2,p3 = points[-3:]
	slope_1 = slope(p1,p2)
	slope_2 = slope(p1,p3)
	angle_tan = abs((slope_1-slope_2)/(1+slope_1*slope_2))
	final_angle = round(math.degrees(math.atan(angle_tan)))
	print(final_angle)



def mouseevent(event,x,y,flags,params):
	if event == cv2.EVENT_LBUTTONDOWN:
		cv2.circle(img,(x,y),5,(0,0,255),cv2.FILLED)
		points.append([x,y])
		


while True:
	if len(points)%3==0 and len(points)!=0:
			find_angle(points)

	cv2.imshow("Image",img)
	cv2.setMouseCallback('Image',mouseevent)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		points = []
		img = cv2.imread(img_path)

cv2.waitKey(0)
cv2.destroyAllWindows()