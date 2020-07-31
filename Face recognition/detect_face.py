import face_recognition
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


all_image = 'all_images'

image = [os.path.join(all_image,img) for img in os.listdir(all_image)]

img = [cv2.imread(i) for i in image]

classes = [os.path.splitext(i)[0] for i in os.listdir(all_image)]

cap = cv2.VideoCapture(0)

#For encodings of folder images
encodings_folder = [face_recognition.face_encodings(i)[0] for i in img]

print("Encoding completed")

while True:
	ret, frame = cap.read()
	location = face_recognition.face_locations(frame)
	encoding = face_recognition.face_encodings(frame,location)

	for all_encoding,all_location in zip(encoding,location):
		matches = face_recognition.compare_faces(encodings_folder,all_encoding)
		distance = face_recognition.face_distance(encodings_folder,all_encoding)
		index = np.argmin(distance)

		if matches[index] and min(distance)<0.5:
			name = classes[index].upper()
			print('Your name is',name,'','and you confidence is',min(distance))
			y1,x2,y2,x1 = all_location
			cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
			cv2.putText(frame,name,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)

	cv2.imshow("Frame",frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

# facial_locations = [face_recognition.face_locations(i) for i in img]

# facial_landmarks = [face_recognition.face_landmarks(i) for i in img]

# print(facial_landmarks)

# test_image = cv2.imread('test.jpg')

# encodings = [face_recognition.face_encodings(i) for i in img]

# test_encodings = face_recognition.face_encodings(test_image)[0]

# face_distance = face_recognition.face_distance()

# result = face_recognition.compare_faces(encodings,test_encodings)




