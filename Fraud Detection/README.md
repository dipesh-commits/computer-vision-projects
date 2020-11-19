# pdf_img_fraud_detect

### Contains the code for detecting fraudulent documents for images


**Input**
Path of image or pdf

**Output**
- Dictionary containing fraudulent status with fraud message

#### CLI for fraud detection on image

```
usage: fraud_detect_image.py [-h] -i INPUT -t THRESHOLD [-d DEBUG]

Get fraudaulent status from image document

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to test image
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold value to compare the result(1,2 or 3 according to edited images)
  -d DEBUG, --debug DEBUG
                        Debugger(default=False)

**Sample code**

`python3 fraud_detect_image.py -i test.jpg -t 1

```

