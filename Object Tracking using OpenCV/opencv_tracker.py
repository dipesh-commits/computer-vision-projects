import cv2
import numpy as np


tracker = cv2.TrackerMOSSE_create()
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

box = (230, 100, 230, 300)

ret = tracker.init(frame,box)



while True:
	ret, frame = cap.read()
	if ret:
		timer = cv2.getTickCount()

		ret, box = tracker.update(frame)

		fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)

		if ret:
			p1 = (int(box[0]),int(box[1]))
			p2 = (int(box[0]+box[2]),int(box[1]+box[3]))
			cv2.rectangle(frame,p1,p2,(255,0,0),2,1)
			cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2);


		else:
			cv2.putText(frame,"Tracking failed",(100,80),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2)

		

		out.write(frame)

		cv2.imshow("image",frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

cap.release()
cv2.destroyAllWindows()

