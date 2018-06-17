import numpy as np
import cv2

img = cv2.imread('obstacle_scene_5.jpg')
dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

lower = np.array([40,40,40])  #-- Lower range --
upper = np.array([110,110,110])  #-- Upper range --
mask = cv2.inRange(dst, lower, upper)
#cv2.imshow('Mask',mask)

mask = 255 - mask
res = cv2.bitwise_and(dst, dst, mask = mask)  #-- Contains pixels having the gray color--

cv2.imshow('Original, Denoised, Masked',np.hstack([img,dst,res]))

cv2.waitKey(0)
cv2.destroyAllWindows()
