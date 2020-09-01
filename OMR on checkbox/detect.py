# This task is to detect whether the checkbox area is checked or not.


# importing the necessary packages
import os
import cv2
import json
import argparse
import numpy as np



def calculate_result(vertical,horizontal,threshold):
	"""
	Calculate final result based on vertical scanning and horizontal scanning of the image and compared with threshold value

	Args:
		vertical(Vertical black pixels):
			Total black pixels based on vertical scanning.
		horizontal(Horizontal black pixels):
			Total black pixels based on horizontal scanning.
		threshold(Value for thresholding):
			Minimum value compared with total pixels.

	Returns:
		True:
			If total pixels are greater than or equal to threshold.
		False:
			If total pixels are less than threshold.
	"""

	# calculating total pixels by adding vertical and horizontal pixels
	totalPixels = vertical+horizontal
	if totalPixels>=threshold:
		return True
	else:
		return False






def vertical_scanning(img,center):	
	"""
	Performing vertical scanning on image across three places after detecting checkbox i.e center-top, halfway through 
	top-left and center-top, halfway through center-top and top-right going all way through these points to bottom of image.

	Args:
		img(Image after detecting checkbox):
			Image after detecting checkbox contours for scanning vertically across three places.
		center(List of three coordinates):
			List of all three coordinates from where vertical scanning is started.

	Returns:
		total_vertical_blackpixels:
			All the black pixels are added after scanning through three coordinates to bottom of image.

	"""
	total_vertical_blackpixels = 0

	#looping through three coordinates going from top to bottom
	for i in center:
		for j in range(0,(img.shape[0]-i[1])):						
			if img[i[1]+j,i[0]]<128:
				total_vertical_blackpixels+=1
	return total_vertical_blackpixels



def horizontal_scanning(img,center):
	"""
	Performing horizontal scanning on image across three places after detecting checkbox i.e center-left, halfway through 
	top-left and center-left, halfway through center-left and bottom-left going all way through these points to right of image.

	Args:
		img(Image after detecting checkbox):
			Image after detecting checkbox contours for scanning horizontally across three places.
		center(List of three coordinates):
			List of all three coordinates from where horizontal scanning is started.

	Returns:
		total_horizontal_blackpixels:
			All the black pixels are added after scanning through three coordinates to right of image.

	"""
	total_horizontal_blackpixels = 0

	#looping through three coordinates going from left to right
	for i in center:
		for j in range(0,img.shape[1]-i[0]):
			if img[i[1],i[0]+j]<128:
				total_horizontal_blackpixels+=1
	return total_horizontal_blackpixels




def apply_transformation(img):
	"""
	Different transformations are applied for successful morphological transformations of image.

	Args:
		img(Image for applying transformation):
			Inverted thresholded image is passed for successful detecting horizontal and vertical lines after thresholding of image.

	Returns:
		finalImageThresholded:
			Thresholded image is returned after detecting vertical and horizontal lines inside the image
	"""

	#determining horizontal and vertical lines for contour detection using vertical and horizontal kernel
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
	scaling_factor = img.shape[1]//3
	vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,img.shape[1]//scaling_factor))
	horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(img.shape[1]//scaling_factor,1))
	vertical_erode = cv2.erode(img,vertical_kernel,iterations=3)
	vertical_dilate = cv2.dilate(vertical_erode,vertical_kernel,iterations=3)
	horizontal_erode = cv2.erode(img,horizontal_kernel,iterations=3)
	horizontal_dilate = cv2.dilate(horizontal_erode,horizontal_kernel,iterations=3)

	#performing thresholding after detecting horizontal and vertical lines
	alpha = 0.5
	beta = 1-alpha
	imageWeighted = cv2.addWeighted(vertical_dilate,alpha,horizontal_dilate,beta,0.0)
	imageWeightedInverted = cv2.bitwise_not(imageWeighted)
	thresh, finalImageThresholded = cv2.threshold(imageWeightedInverted,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return finalImageThresholded




def write_to_file(dictionary):
	"""
	Writing response to a text file

	Args:
		dictionary(a dictionary with results):
			File consists of dictionary where the key is image file name and the value is result of image.
	"""
	with open('output.txt','w') as f:
		f.write(json.dumps(dictionary))
	



def main():
	"""
	Main function for reading image and and determining contours for checkbox detection. The processed image is passed
	to other function for final detection.

	"""

	# initializing dictionary for attaching response
	output_dictionary = {}


	# looping through each image
	for img in img_path:
		image = cv2.imread(img)
		imageOriginal = image.copy()
		imageGray = cv2.cvtColor(imageOriginal,cv2.COLOR_BGR2GRAY)

		# performing otsu thresholding for gray image
		ret, imageThresholded = cv2.threshold(imageGray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		# inverting thresholded image
		imageInverted = 255-imageThresholded

		
		# passing inverted thresholded image to apply_transformation function and assigning the returned value to final_image_thresh variable
		final_image_thresholded = apply_transformation(imageInverted)

		# extracting basename for image name
		basename = os.path.basename(img)

		# determining contours apply applying transformation over the image and assigning contours value to contours variable
		contours, _ = cv2.findContours(final_image_thresholded,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		if not contours:
			continue
		else:
			max_contour = max(contours,key=cv2.contourArea)

			# determining the bounding box for maximum contour's area present in the image which will be rectangular checkbox for maximum condition
			x1, y1, w1, h1 = cv2.boundingRect(max_contour)
			if (w1 > 4 and h1 > 2):
				x2,y2 = x1+w1,y1+h1

				# determining padding for getting area inside the checkbox
				w_padding = int(w1*0.3)
				h_padding = int(h1*0.35)

				# cropping the image for determining whether black pixels is present inside the checkbox or not
				imageCropped = imageThresholded[y1+h_padding:y2-h_padding, x1+w_padding:x2-w_padding]

				# image is resized to 24*24 pixels for uniform size
				imageResized = cv2.resize(imageCropped,(24,24))
				x,y = 0,0
				h,w = imageResized.shape


				# finding three coordinates from resized image for vertical scanning process
				# applying vertical scanning function which return number of black pixels present inside the box
				center_x = [int(0.5*(x+w)),y]
				center_x_left = [int(0.25*(x+w)),y]
				center_x_right = [int(0.75*(x+w)),y]
				all_vertical_list = [center_x,center_x_left,center_x_right]
				total_vertical_blackpixels = vertical_scanning(imageResized,all_vertical_list)


				# finding three coordinates from resized image for horizontal scanning process
				# applying horizontal scanning function which return number of black pixels present inside the box
				center_y = [x,int(0.5*(y+h))]
				center_y_top = [x,int(0.25*(y+h))]
				center_y_bottom = [x,int((0.75*(y+h)))]
				all_horizontal_list = [center_y,center_y_top,center_y_bottom]
				total_horizontal_blackpixels = horizontal_scanning(imageResized,all_horizontal_list)

				# calling calculate_result function for returning final response and response is assigned to result variable
				result = calculate_result(total_vertical_blackpixels,total_horizontal_blackpixels,10)

				# Attaching image name as key and result as value to dictionary and printing response
				output_dictionary[str(basename)] = str(result)
				print(f"Result of {str(basename)}: {str(result)}")

	# calling write_to_file function for writing to the file
	write_to_file(output_dictionary)



if __name__ == '__main__':

	# constructing argument parser and parse the argument to pass the test folder for detection
	ap = argparse.ArgumentParser()
	ap.add_argument("-i","--input", required=True, help="Path to test folder")
	args = vars(ap.parse_args())

	# folder containing images
	data_folder = args["input"]

	# passing the folder containing images inside it and finding image path
	img_path = [os.path.join(data_folder,img) for img in os.listdir(data_folder)]

	# calling main function
	main()





