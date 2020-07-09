import cv2
import numpy as np

weights_file = 'yolov3.weights'
cfg_file = 'yolov3.cfg'
class_file = 'coco.names'

classes=[]

net = cv2.dnn.readNet(weights_file,cfg_file)

with open(class_file,'r') as f:
	classes = f.read().splitlines()


img = cv2.imread('test.jpg')
img = cv2.resize(img,None,fx=0.6,fy=0.6)
height, width, channels = img.shape

boxes=[]
confidences=[]
class_idxs=[]

blob = cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True, crop=False)

net.setInput(blob)
output_layers = net.getUnconnectedOutLayersNames()


layersOutputs = net.forward(output_layers)

for output in layersOutputs:
	for detection in output:
		scores = detection[5:]
		class_id = np.argmax(scores)
		confidence = scores[class_id]

		if confidence>0.5:
			center_x = int(detection[0]*width)
			center_y = int(detection[1]*height)
			w = int(detection[2]*width)
			h = int(detection[3]*height)

			x = int(center_x-w/2)
			y = int(center_y-h/2)

			boxes.append([x,y,w,h])
			confidences.append(float(confidence))
			class_idxs.append(class_id)


indices = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)

FONT = cv2.FONT_HERSHEY_PLAIN
COLOR = np.random.uniform(0,255,size=(len(boxes),3))

for i in indices.flatten():
	x,y,w,h = boxes[i]
	label = str(classes[class_idxs[i]])
	confidence = str(round(confidences[i],2))
	color = COLOR[i]

	cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
	cv2.putText(img,label,(x,y),FONT,2,(255,255,255),2)



cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
