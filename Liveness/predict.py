import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import pickle

detector = MTCNN()



classes = ['Real', 'Fake']

filename = 'finalized_model_RandomForest_norm.sav'
loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)


cap = cv2.VideoCapture(0)
tolerance = 1e-7

def get_pixel_value(img, center, x, y):
	new_value = 0
	try:
		if img[x][y] >= center:
			new_value = 1
	except:
		pass
	return new_value


def lbp_pixel(img,x,y):
	center = img[x][y]
	val_ar = []
	val_ar.append(get_pixel_value(img,center,x-1,y-1))
	val_ar.append(get_pixel_value(img,center,x-1,y))
	val_ar.append(get_pixel_value(img,center,x-1,y+1))
	val_ar.append(get_pixel_value(img,center,x,y+1))
	val_ar.append(get_pixel_value(img,center,x+1,y+1))
	val_ar.append(get_pixel_value(img,center,x+1,y))
	val_ar.append(get_pixel_value(img,center,x+1,y-1))
	val_ar.append(get_pixel_value(img,center,x,y-1))


	bin_to_dec = [1,2,4,8,16,32,64,128]

	val = 0
	for i in range(len(val_ar)):
		val+= val_ar[i]*bin_to_dec[i]
	return val

while(True):

	ret, frame = cap.read()


	if ret:
		detections = detector.detect_faces(frame)
		for face in detections:
			if face['confidence'] > 0.90:
				x, y , w, h = face['box']

				required = frame[y:y+h,x:x+w]

				required = cv2.resize(required,(64,64))
				img_gray = cv2.cvtColor(required,cv2.COLOR_BGR2GRAY)


				img_lbp = np.zeros((64,64,3),np.uint8)



				for i in range(0,64):
					for j in range(0,64):
						img_lbp[i][j] = lbp_pixel(img_gray,i,j)

				hist, bin_edges = np.histogram(img_lbp.ravel(),density=True)
				hist = hist.astype("float")
				hist/= (hist.sum() + tolerance)



				hist = hist.reshape(1,-1)

			



				# img_lbp = img_lbp.reshape(1,64,64,3)

				prediction = loaded_model.predict(hist)[0]
				class_probabilities = loaded_model.predict_proba(hist)
				index = np.argmax(class_probabilities)
				print(class_probabilities)

				if class_probabilities[0][index]>0.6:

					label = classes[index]



				# predicted_class = classes[index]

					cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
					cv2.putText(frame,"Prediction"+ ":"+ str(label),(x,y-10),cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),2)

					
		cv2.imshow("Frame",frame)


		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

cap.release()

cv2.destroyAllWindows()