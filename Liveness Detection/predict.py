import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras.models import load_model

detector = MTCNN()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

classes = ['fake', 'real']

model = load_model('model.h5')

cap = cv2.VideoCapture(0)

while(True):

	ret, frame = cap.read()



	if ret:
		detections = detector.detect_faces(frame)
		for face in detections:
			if face['confidence'] > 0.95:
				x, y , w, h = face['box']
				required = frame[y:y+h,x:x+w]

				required = cv2.resize(required,(360,360))

				final_pic = required.reshape(1,360,360,3)

				prediction = model.predict(final_pic)


				index = np.argmax(prediction)

				predicted_class = classes[index]

				cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
				cv2.putText(frame,"Prediction"+ ":"+ predicted_class,(x,y-10),cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),2)

					# out.write(frame)
		cv2.imshow("Frame",frame)


		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

cap.release()

cv2.destroyAllWindows()