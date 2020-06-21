#importing necessary libraries
import cv2
import numpy as np




#Video capturing

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)    #converting rgb image to hsv for masking


	#assigning value for masking purpose for red channel
	low_red = np.array([5,120,120])
	high_red = np.array([20,255,255])					

	red_mask = cv2.inRange(img,low_red,high_red)

	final_red_img = cv2.bitwise_and(frame,frame,mask=red_mask)





	#for green channel
	low_green = np.array([40,40,40])
	high_green = np.array([70,255,255])

	green_mask = cv2.inRange(img,low_green,high_green)

	final_green_img = cv2.bitwise_and(frame,frame,mask=green_mask)




	#for blue channel
	low_blue = np.array([110,50,50])
	high_blue = np.array([130,255,255])

	blue_mask = cv2.inRange(img,low_blue,high_blue)


	final_blue_img = cv2.bitwise_and(frame,frame,mask=blue_mask)




	#determining countours for real time detection


	#for red channel
	contours, hierarchy = cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if area>2000:
			x,y,w,h = cv2.boundingRect(contour)
			frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)
			cv2.putText(frame,"Red color found",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255))
			print(area)


	#for green channel
	contours, hierarchy = cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if area>2000:
			x,y,w,h = cv2.boundingRect(contour)
			frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
			cv2.putText(frame,"Green color found",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0))
			print(area)

	#for blue channel
	contours, hierarchy = cv2.findContours(blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if area>2000:
			x,y,w,h = cv2.boundingRect(contour)
			frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)
			cv2.putText(frame,"Blue color found",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0))			
			print(area)

	#showing respective frame
	cv2.imshow('myimg',frame)
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

