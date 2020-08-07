import cv2
import os
import numpy as np
from mtcnn.mtcnn import MTCNN





output_path = 'dataset/real'

dataset = 'dataset'   #data path
detector = MTCNN()
# cap = cv2.VideoCapture('video/real_video.mp4')

read = 0

#for detecting face in images
image_path = []
for i in os.listdir(dataset):
	image_path.append(os.path.join(dataset,i))

try:
	for j in image_path:
		image = cv2.imread(j)
		faces = detector.detect_faces(image)
		read+=1
		for face in faces:
			x, y ,w, h = face['box']
			required = image[y:y+h,x:x+w]
			saving_path = os.path.join(output_path,"{}.png".format(read))
			cv2.imwrite(saving_path,required)
			print("[INFO] saved {} to disk".format(saving_path))
except:
	pass


#for detecting in videos
# while True:
# 	ret, frame = cap.read()

# 	if not ret:
# 		break

# 	else:
# 		read+=1
# 		height, width, channels = frame.shape

# 		frame = cv2.resize(frame,(720,640),fx=0,fy=0)

# 		frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
# 		saving_path = os.path.join(output_path,"{}.png".format(read))
# 		cv2.imwrite(saving_path,frame)
# 		print("[INFO] saved {} to disk".format(saving_path))

			# cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
			# cv2.imshow("Frame",frame)


			# if cv2.waitKey(1) & 0xFF == ord('q'):
			# 	break


print("Total frames",read)
# cap.release()
cv2.destroyAllWindows()
