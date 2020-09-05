import cv2
import numpy as np
import time


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

cap = cv2.VideoCapture(0)

time.sleep(0)
count = 0
background = 0

for i in range(60):
	ret, background = cap.read()
background = np.flip(background,axis=1)

while(cap.isOpened()):
	ret, frame = cap.read()
	if not ret:
		break
	count+=1
	frame = np.flip(frame,axis=1)

	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	lower_red = np.array([0,125,50])
	higher_red = np.array([10,255,255])

	mask1 = cv2.inRange(hsv,lower_red,higher_red)

	lower_red = np.array([170,120,70])
	upper_red = np.array([180,255,255])

	mask2 = cv2.inRange(hsv,lower_red,upper_red)

	mask = mask1+mask2

	mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
	mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))

	mask3 = cv2.bitwise_not(mask)

	res1 = cv2.bitwise_and(frame,frame,mask=mask3)

	res2 = cv2.bitwise_and(background,background,mask=mask)

	finalOutput = cv2.addWeighted(res1,1,res2,1,0)

	out.write(finalOutput)
	cv2.imshow("Image",finalOutput)

	if cv2.waitKey(1) & 0xFF==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
