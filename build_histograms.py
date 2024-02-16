import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('./image_test/1636738357390407600.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
x, y, w, h = 321,269,123,189
mask[y:y+h, x:x+w] = 255
masked_img = cv.bitwise_and(img,img,mask = mask)
# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv.calcHist([img],[0],mask,[256],[0,256])
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])
plt.show()


# have to do 4 total comparaison of a person box with box, etc...
# image 1 test boxes
# person 1: 232,128,70,269
# person 1 upper half: 232, 128, 70, 135
# person 2: 375,271,156,188
# person 2: upper half: 375, 271, 156, 94
# person 3: 375,271,156,188
# person 3 upper half: 375, 271, 156, 94

#image 2 test boxes
# person 1: 463, 251, 112, 206
# person 1 upper half: 463, 251, 112, 103
# person 2: 321,269,123,189
# person 2 upper half: 321,269,123,95