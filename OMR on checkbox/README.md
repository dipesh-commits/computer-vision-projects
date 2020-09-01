# Detect whether the checkbox area is checked or not

Requirements:
1. Python
2. Opencv
3. Numpy


Usage: 

To run the script, run the following command in terminal:

	python detect.py -i <image Folder>


Approach:

The following methods are performed to detect whether the checkbox is checked or not:

1. Read image from the folder and convert it into grayscale.
2. Apply thresholding to the grayscale image.
3. Find horizontal and vertical lines of checkbox by using horizontal and vertical kernels.
4. Determine the contour of the checkbox from horizontal and vertical lines.
5. Cropped and resize the thresholded image using contour value.
6. Apply vertical and horizontal scanning over the cropped image across three points in vertical and horizontal detections.
7. The three points for vertical scanning are determined by dividing top horizontal line into four parts equally and three points are chosen except top-left and top-right.
8. The three points for horizontal scanning are determined by dividing left vertical line into four parts equally and three points are chosen except top-left and bottom-left.
9. Loop through each points across vertical line all way from horizontal top to horizontal bottom and across horizontal line all way from left to right and calculate number of black pixels containing in it.
10. If the checkbox is empty then it doesn't contain any black pixels and if the checkbox is marked then it contains some black pixels.
11. Finally, the number of black pixels are compared with threshold value and response is returned as True and False.










