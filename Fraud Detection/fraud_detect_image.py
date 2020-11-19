""" Fraud detection on document images """

# importing necessary libraries
import os,math
import argparse
import exifread
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance, ImageFilter



def check_metadata(imgpath:str) -> dict:
	"""
	Helps to check whether given image file is fraud or not. Check image metadata
	whether it has been changed or not. 

	Args:
		imgpath(image path):
				 path of image to check the status of fraudulent

	Returns:
		result(json result):
				json values regarding fraud status

	"""

	# initializing dictionary to append metadata
	metadata_result = {}

	# read image file to extract metadata
	f = open(imgpath,'rb')
	tags = exifread.process_file(f)
	# check if any tags exist or not
	if tags:
		for tag in tags.keys():
			metadata_result[tag] = tags[tag]

		# check if original datetime is equal to image date time. If image is edited, then they are not equal
		if 'EXIF DateTimeOriginal' in metadata_result:
			if metadata_result['Image DateTime']!=metadata_result['EXIF DateTimeOriginal']:
				return {'response':'Fraud','message':'Original DateTime is different from Image DateTime'}
				# print(f"The file is forged")
				# sys.exit()

		# check if any third party is used for editing images
		elif 'Image Software' in metadata_result:
			return {'response':'Fraud','message':'Image is edited using image editing tools'}


		# perform error level if any of the above doesn't work
		else:
			result = perform_ela(imgpath)
			return result

	# if any tags are not found, then perform error level analysis
	else:
		result = perform_ela(imgpath)
		return result
		



def perform_ela(imgpath:str,debug:bool=False) -> dict:
	"""
	Helps to perform error level analysis to detect copy-move
	forgery inside documents. 

	Args:
		imgpath(image path):
			path of image to check the status of fraudulent
		debug(debugger):
			debugger tool to visualize result

	Returns:
		result(json result):
				json values regarding fraud status

	"""

	# temporary compressed image
	TEMP = 'temp.jpeg'
	threshold= 150
	compare_result= int(args['threshold'])
	debug_status = args['debug']


	# load Original Image 
	original = Image.open(imgpath)
	img_original = original.copy()

	if original is None:
		return {'message':'Couldnt process your request.'}

	# converting to Black and White 
	original_gray = original.convert('L')
	original_gray = ImageEnhance.Sharpness(original_gray).enhance(0.0)

	# compressing at 90% 
	original_gray.save(TEMP,'JPEG',quality=90)
	temporary = Image.open(TEMP)

	# find difference between original and temporary
	diff = ImageChops.difference(original_gray, temporary)

	# find the max value of color band in image
	extrema = diff.getextrema()
	max_diff = extrema[1]
	scale = 255.0/max_diff

	# enhance the image based on that scale
	diff = ImageEnhance.Brightness(diff).enhance(scale)

	if debug_status:
		plt.title("Differencing original and temporary image")
		plt.imshow(diff)
		plt.show()
		

	# fetch the histogram of the difference image (Count of color pixels)
	lists = diff.histogram(mask=None, extrema=None)
	if debug_status:
		plt.title("Histogram of difference between original and temporary image")
		plt.xlabel("pixels")
		plt.ylabel("Total number")
		for i in range(0, 256):
			plt.bar(i, lists[i], color = 'b',alpha=1.0)
		plt.show()

	# calculate Threshold by keeping last 75 pixels
	pixels = 0
	for i in range(255,1,-1):
		if pixels+lists[i] <= 75:
			pixels += lists[i]
		else:
			threshold = i+1
			break
	# apply Threshold
	bw = diff.point(lambda x: 0 if x < threshold else 255, '1')   #'1' because source image has mode 'L'

	if debug_status:
		plt.title(f"Image After applying threshold of {threshold}")
		plt.xlabel('Width')
		plt.ylabel('Height')
		plt.imshow(bw)
		plt.show()
		

	# calculate Radius
	w, h = bw.size
	radius = int(math.sqrt((w*h)/(3.14*625.23)))

	# maintain a pixel array (Pixel Number : X-Y Co-ordinate)
	coordinates = []

	# edges {(V1->V2) (V2->V3) (V4->V5)}
	edges = []

	# scan the entire image and fetch co-ordinates of white pixels
	bwa = bw.load()
	for x in range(w):
		for y in range(h):

			# fetch each pixel 
			color = bwa[x,y]
		
			# if pixel is white, record its co-ordinates
			if color==255:
				coordinates.append([x,y])


	# if debug_status:
	# 	for i in coordinates:
	# 		x = i[0]
	# 		y = i[1]
	# 		print(x,y)
	# 		cv2.circle(np.float32(img_original),(x,y),10,(0,0,0),5)
	# 	plt.imshow(img_original)
	# 	plt.show()

	# loop through XY Co-Ordinates and find edges 
	for coord in coordinates:

		index = coordinates.index(coord)
		
		x1 = coord[0]
		y1 = coord[1]

		for next_index in range(index+1,len(coordinates)):
			
			x2 = coordinates[next_index][0]
			y2 = coordinates[next_index][1]

			distance = math.sqrt(((x1-x2)**2)+((y1-y2)**2))

			if (distance < (2*radius)):
				edges.append([index,next_index])

	# create a list that has connections for every pixel (V1:V2,V3,...) (V2:V1,V2,...)
	connectedPixelsList = []

	# no of white pixels
	total_white_pixels = len(coordinates)
	connectedPixelsList = getConnectedPixels(edges,total_white_pixels)

	# labels of clusters (Starting value -> 100)
	labelCount = 100

	# dictionary (Pixel:Label)
	label_of_pixels = {}

	# assign every pixel a label of 0 
	for index in range(0,len(connectedPixelsList)):
		label_of_pixels.update({connectedPixelsList[index][0]:0})


	# check neighbor of every pixel and find the root label of every pixel
	for index in range(0,len(connectedPixelsList)):
		element = connectedPixelsList[index]

		# arbitrary root number
		root = 1000

		# find label with lowest value and assign to root
		for i in range(1,len(element)):
			pixel = element[i]
			label = label_of_pixels.get(pixel)

			if label > 0 and root > label:
				root = label

		# Union-Find Algorithm 
		
		# if no root found, assign an arbitrary label
		if root == 1000:
			labelCount += 1
			label_of_pixels.update({element[0]:labelCount})
		else:
			# update all pixels with root as label 
			for i in range(0,len(element)):
				pixel = element[i]
				label_of_pixels.update({pixel:root})


	# count the number of white pixels for each label
	color_count = {}
	for lp in label_of_pixels.items():
		label = lp[1]
		if label not in color_count:
			color_count.update({label:1})
		else:
			color_count.update({label:color_count.get(label)+1})
	# white Pixels in each Cluster 
	max = 0
	for p in color_count.values():
		if max < p:
			max = p
	
	if compare_result==1:
		compare=10
	elif compare_result==2:
		compare=30
	elif compare_result==3:
		compare=60
	else:
		return {"Give proper value for threshold value to compare results i.e 1, 2 or 3"}


	maxToTotalRatio = (max/total_white_pixels)*100
	print(maxToTotalRatio)
	if maxToTotalRatio >= compare:
		return {'response':'Fraud','message':'Copy-move forgery is detected'}
	else:
		return {'response':'Not Fraud','message':'No any fraud detected'}




def getConnectedPixels(edges:list,total_white_pixels:int) -> list:
	"""
	Helps to find connections for each pixels

	Args:
		edges: edges after looping through XY coordinates
		total_white_pixels: total number of white pixels for each pixels

	Returns:
		cpl: all connected pixels labeling
	"""

	cpl = []

	for index in range(0,total_white_pixels):

		# create new element for new pixel
		cpl.append([index])

		# index of the last element of list
		cpl_index = (len(cpl) - 1)

		for i in range(0,len(edges)):

			edge = edges[i]
			start = edge[0]
			end = edge[1]

			# edges (V1->V2) where {start: V1, end: V2}
			if start == index:
				cpl[cpl_index].append(end)
				continue
			elif end == index:
				cpl[cpl_index].append(start)
				continue

		# remove all pixels with no connections
		if len(cpl[cpl_index]) == 1:
			cpl.remove(cpl[cpl_index])

	return cpl


if __name__ == "__main__":

	# CLI for passing image path
	ap = argparse.ArgumentParser(prog="Fraud detection",description="Process display argument")
	ap.add_argument("-i","--input",required=True,type=str,help="Path to test image")
	ap.add_argument("-t","--threshold",required=True,type=int,help="Threshold value to compare the result")
	ap.add_argument("-d","--debug",help="Debugger",type=bool,default=False)
	args = vars(ap.parse_args())

	# getting image path from CLI
	img_path = args["input"]
	compare_result= int(args['threshold'])

	# calling function for checking fraud status
	final_response = check_metadata(img_path)
	print(final_response)
