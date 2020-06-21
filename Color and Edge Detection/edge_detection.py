import cv2
import numpy as np


cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object 
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter('color_detection.avi', fourcc, 20.0, (820, 640)) 

while True:
	ret, frame = cap.read()
	img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	edges = cv2.Canny(img,40,80)

	final_img = np.hstack((img,edges))


	out.write(final_img) 
	cv2.imshow('Edge',final_img)

	if cv2.waitKey(1) & 0xFF==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

