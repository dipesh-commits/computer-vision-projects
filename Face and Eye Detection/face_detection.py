import cv2
import numpy as np
import matplotlib.pyplot as plt



cap = cv2.VideoCapture(0)

while(True):
	ret, img = cap.read()


	face_haar_cascades = cv2.CascadeClassifier('lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
	eye_haar_cascades = cv2.CascadeClassifier('lib/python3.6/site-packages/cv2/data/haarcascade_eye.xml')

	coordinates_face = face_haar_cascades.detectMultiScale(img,scaleFactor=1.2,minNeighbors=5)




	for (x,y,w,h) in coordinates_face:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
		roi = img[y:y+h,x:x+w]
		coordinates_eye = eye_haar_cascades.detectMultiScale(roi)

		for (ex,ey,ew,eh) in coordinates_eye:
			cv2.rectangle(roi,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

	cv2.namedWindow('image',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('image', 600,600)
	cv2.imshow('image',img)
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break


cap.release()
cv2.destroyAllWindows()