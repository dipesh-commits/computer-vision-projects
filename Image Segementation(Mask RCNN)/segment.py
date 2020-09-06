import cv2
import numpy as np



confidence_threshold = 0.5
segment_threshold = 0.3

video_file = 'aa.mp4'

weights_file = 'mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
labels_file = 'coco.names'
color_file = 'color.txt'
config_file = 'mask_rcnn_inception_v2_coco_2018_01_28/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'


with open(labels_file,'r') as f:
	classes = f.read().split('\n')


with open(color_file,'r') as f:
	intial_color = f.read().split('\n')

all_colors = list()
for i in range(len(intial_color)):
	rgb_color = intial_color[i].split(' ')
	color = np.array([float(rgb_color[0]), float(rgb_color[1]), float(rgb_color[2])])

	all_colors.append(color)







net = cv2.dnn.readNetFromTensorflow(weights_file,config_file)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
cap = cv2.VideoCapture(video_file)

while True:
	ret, frame = cap.read()

	if ret:
		blob = cv2.dnn.blobFromImage(frame,swapRB=True,crop=False)
		net.setInput(blob)
		boxes, segments = net.forward(['detection_out_final','detection_masks'])


		# postprocess(boxes,segments)

		total_classes = segments.shape[1]
		total_detection = boxes.shape[2]

		H = frame.shape[0]
		W = frame.shape[1]

		for b in range(total_detection):
			box = boxes[0,0,b]
			segment = segments[b]
			score = box[2]
			if score>confidence_threshold:
				class_id = int(box[1])

				x = int(box[3]*W)
				y = int(box[4]*H)
				w = int(box[5]*W)
				h = int(box[6]*H)

				x = max(0,min(x,W-1))
				y = max(0,min(y,H-1))
				w = max(0,min(w,W-1))
				h = max(0,min(h,H-1))

				class_segment = segment[class_id]

				# drawBox(frame, class_id, score, x, y, w, h, class_segment)

				cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
				label = classes[class_id]
				
				cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

				class_segment = cv2.resize(class_segment, (w - x + 1, h - y + 1))
				segment = (class_segment > segment_threshold)

				roi = frame[y:h+1, x:w+1][segment]

				color = all_colors[class_id%len(all_colors)]

				frame[y:h+1, x:w+1][segment] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.7 * roi).astype(np.uint8)
				segment = segment.astype(np.uint8)
				contours, hierarchy = cv2.findContours(segment,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
				cv2.drawContours(frame[y:h+1, x:w+1], contours, -1, color, 3, cv2.LINE_8, hierarchy, 100)







		t, _ = net.getPerfProfile()

		label = 'Mask-RCNN : Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())

		cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

		out.write(frame)
		cv2.imshow("Image Segmentation",frame)

		if cv2.waitKey(1) | 0xFF == ord('q'):
			break

	else: 
		break

out.release()
cap.release()
cv2.destroyAllWindows()


