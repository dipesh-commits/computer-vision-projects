import cv2
import os
import numpy as np
from keras.models import load_model


my_model = load_model('hand_model.h5')

train_data_path = 'hand_gesture/dataset/training_set'

label = []
for folder in os.listdir(train_data_path):
	label.append(folder)



print(label)

cap = cv2.VideoCapture(0)

while(True):
	ret, img = cap.read()
	if ret:
		image_copy = img.copy()
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		thresh_value = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,229,7)
		# final_img = cv2.bitwise_and(image_copy,image_copy,mask=thresh_value)

		thres_value_3 = cv2.cvtColor(thresh_value,cv2.COLOR_GRAY2BGR)




		part = cv2.rectangle(image_copy,(200,200),(200+250,200+250),(255,0,0),3)

		required_part = thres_value_3[200:450,200:450]
		print(required_part.shape)

		required_part = cv2.resize(required_part,(90,70),interpolation=cv2.INTER_AREA)
		# final_img = cv2.cvtColor(final_img,cv2.COLOR_GRAY2BGR)
		required_part = required_part.reshape(1,70,90,3)
		prediction = my_model.predict(required_part)
		my_predict = np.argmax(prediction)
		cv2.putText(image_copy,label[my_predict],(200,190),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255))

		# print(label[my_predict])

		all_videos = np.hstack((image_copy,thres_value_3))
		cv2.imshow('My Img',all_videos)
		if cv2.waitKey(1) & 0xFF==ord('q'):
			break

cap.release()
cv2.destroyAllWindows()



