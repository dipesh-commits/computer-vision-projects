import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from skimage import feature

import pickle

detector = MTCNN()


# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('demo_hog.avi',fourcc, 20.0, (640,480))

# classes = ['Real', 'Fake']

filename = 'finalized_model_Gradientboosting.sav'
loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)


cap = cv2.VideoCapture('next.mp4')
tolerance = 1e-7

def get_lbp(image,points,radius,tolerance):
	lbp = feature.local_binary_pattern(image,points,radius,method="uniform")
	hist, bins = np.histogram(lbp.ravel(),bins=np.arange(0,points + 3),range=(0,points + 2))
	hist = hist.astype("float")
	hist/= (hist.sum()+tolerance)

	return hist

# cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions

while(True):

	ret, frame = cap.read()


	if ret:
		frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
		frame = cv2.resize(frame, (480,740))                    # Resize image

		detections = detector.detect_faces(frame)
		for face in detections:
			if face['confidence'] > 0.80:
				x, y , w, h = face['box']

				required = frame[y:y+h,x:x+w]

				required = cv2.resize(required,(200,200))
				img_gray = cv2.cvtColor(required,cv2.COLOR_BGR2GRAY)



				hist = get_lbp(img_gray,24,8,1e-7)



				hist = hist.reshape(1,-1)

			



				# img_lbp = img_lbp.reshape(1,64,64,3)

				prediction = loaded_model.predict(hist)[0]
				print(prediction)
				# class_probabilities = loaded_model.predict_proba(hist)
				# index = np.argmax(class_probabilities)
				# print(class_probabilities)

				# if class_probabilities[0][index]>0.6:

				# 	label = classes[index]



				# predicted_class = classes[index]

				cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
				cv2.putText(frame,"Prediction"+ ":"+ str(prediction),(x,y-10),cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),2)

		# out.write(frame)
		cv2.imshow("Frame",frame)


		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

cap.release()

cv2.destroyAllWindows()