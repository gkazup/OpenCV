# USAGE
# python edge_detect.py --image page.jpg 

# import the necessary packages
from pyimagesearch.transform import four_point_transform
from pyimagesearch import imutils
from skimage.filters import threshold_adaptive
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

denoised = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)

image = cv2.GaussianBlur(image, (5, 5), 0)
edged = cv2.Canny(denoised, 10, 70)

#gray = np.float32(gray)
#harris = cv2.cornerHarris(gray,2,3,0.04)
#harris = cv2.dilate(harris,None)

# show the original image and the edge detected image
print "STEP 1: Edge Detection"
cv2.imshow("Image", image)
#cv2.imshow("Harris", harris)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
