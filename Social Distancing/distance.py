import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt

weights_file = 'yolov3.weights'
conf_file = 'yolov3.cfg'
label_file = 'coco.names'

# with open(label_file) as f:
#     classes = f.read().split('\n')

classes= open(label_file).read().strip().split("\n")  


def check_distance(a,b):
	distance = ((a[0] - b[0])**2 + (a[1]-b[1])**2)**0.5
	calibrate = (a[1]+b[1])/2
	if 0<distance<0.25*calibrate:
		return True
	else:
		return False




img = cv2.imread('try.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# img = imutils.resize(img,width=500)

net = cv2.dnn.readNet(weights_file,conf_file)

height, width, channels = img.shape


blob = cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True,crop=False)

net.setInput(blob)

output_layers = net.getUnconnectedOutLayersNames()

layers = net.forward(output_layers)

boxes = []
confidences = []
class_ids = []

for layer in layers:
	for detection in layer:
		scores = detection[5:]
		class_id = np.argmax(scores)
		confidence = scores[class_id]

		if classes[class_id] == 'person':

			if confidence > 0.5:

				center_x = int(detection[0]*width)
				center_y = int(detection[1]*height)
				w = int(detection[2]*width)
				h = int(detection[3]*height)

				x = int(center_x-w/2)
				y = int(center_y-h/2)

				boxes.append([x,y,w,h])
				confidences.append(float(confidence))
				class_ids.append(class_id)

        
indices = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)


FONT = cv2.FONT_HERSHEY_PLAIN
COLOR = np.random.uniform(0,255,size=(len(boxes),3))


if len(indices)>0:
	pairs = []
	status = []
	center = []
	for i in indices.flatten():
		(x,y) = (boxes[i][0], boxes[i][1])
		(w,h) = (boxes[i][2], boxes[i][3])

		center.append([int(x+w/2),int(y+h/2)])
		status.append(False)

	for i in range(len(center)):
		for j in range(len(center)):
			check = check_distance(center[i],center[j])
			if check:
				pairs.append([center[i],center[j]])
				status[i]= True
				status[j] = True


	index = 0








	for i in indices.flatten():
		(x,y) = (boxes[i][0], boxes[i][1])
		(w,h) = (boxes[i][2], boxes[i][3])
		#label = str(classes[class_ids[i]])
		# confidence = str(round(confidences[i],2))
		# color = COLOR[i]
		if status[index] == True:
			cv2.circle(img,(int(x+w/2),y),2,(255,0,0),2)
			# cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			# cv2.putText(img,"Violation",(x,y),FONT,2,(0,0,0),2)

		elif status[index] == False:
			cv2.circle(img,(int(x+w/2),y),2,(0,255,0),2)
			# cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
			# cv2.putText(img,"Safe",(x,y),FONT,2,(0,0,0),2)

		index+=1

	for h in pairs:
		cv2.line(img, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)




    # cv2.putText(img,label,(x,y),FONT,2,(255,255,255),2)

# cv2.imshow('Image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.imshow(img)
plt.show()