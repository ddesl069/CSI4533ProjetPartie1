import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Define a function to calculate histogram correlation
def calculate_correlation(hist1, hist2):
    return cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)

img = cv.imread('./image_test/1636738315284889400.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
# create first mask full box
mask = np.zeros(img.shape[:2], np.uint8)
x, y, w, h = 232,128,70,269
mask[y:y+h, x:x+w] = 255
masked_img = cv.bitwise_and(img,img,mask = mask)
# create second mask height half
mask2 = np.zeros(img.shape[:2], np.uint8)
h2 = h//2
mask2[y:y+h2, x:x+w] = 255
masked_img2 = cv.bitwise_and(img,img,mask=mask2)
# Calculate histogram with mask and without mask
# Check third argument for mask
hist_mask = cv.calcHist([img],[0],mask,[256],[0,256])
hist_mask_normalized = (hist_mask / hist_mask.sum())*100
hist_mask2 = cv.calcHist([img],[0],mask2,[256],[0,256])
hist_mask2_normalized = (hist_mask2/ hist_mask2.sum())*100



img2 = cv.imread('./image_test/1636738357390407600.png', cv.IMREAD_GRAYSCALE)
assert img2 is not None, "file could not be read, check with os.path.exists()"
# create first mask full box
mask_secondimg = np.zeros(img.shape[:2], np.uint8)
x, y, w, h = 463,251,112,206
mask_secondimg[y:y+h, x:x+w] = 255
masked_img3 = cv.bitwise_and(img,img,mask = mask_secondimg)
# create second mask height half
mask_secondimg2 = np.zeros(img.shape[:2], np.uint8)
h2 = h//2
mask_secondimg2[y:y+h2, x:x+w] = 255
masked_img4 = cv.bitwise_and(img,img,mask=mask_secondimg2)
# Calculate histogram with mask and without mask
# Check third argument for mask
hist_mask_secondimg = cv.calcHist([img],[0],mask_secondimg,[256],[0,256])
hist_mask_normalized_secondimg = (hist_mask_secondimg / hist_mask_secondimg.sum())*100
hist_mask2_second_img = cv.calcHist([img],[0],mask_secondimg2,[256],[0,256])
hist_mask2_normalized_secondimg = (hist_mask2_second_img/ hist_mask2_second_img.sum())*100


# Calculate correlations
correlation_1_2_1 = calculate_correlation(hist_mask_normalized, hist_mask_normalized_secondimg)
correlation_1_2_2 = calculate_correlation(hist_mask_normalized, hist_mask2_normalized_secondimg)
correlation_2_2_1 = calculate_correlation(hist_mask2_normalized, hist_mask_normalized_secondimg)
correlation_2_2_2 = calculate_correlation(hist_mask2_normalized, hist_mask2_normalized_secondimg)

# Determine the most accurate comparison
correlations = [correlation_1_2_1, correlation_1_2_2, correlation_2_2_1, correlation_2_2_2]
comparison_names = ["Img 1 Mask 1 vs Img 2 Mask 1", "Img 1 Mask 1 vs Img 2 Mask 2",
                    "Img 1 Mask 2 vs Img 2 Mask 1", "Img 1 Mask 2 vs Img 2 Mask 2"]
max_correlation_index = np.argmax(correlations)
most_accurate_comparison = comparison_names[max_correlation_index]

print(f"The most accurate comparison is: {most_accurate_comparison} with a correlation of {correlations[max_correlation_index]:.4f}")


#display
fig, axs = plt.subplots(2, 2, figsize=(20, 12))

# Comparison 1: Image 1 Mask 1 with Image 2 Mask 1
axs[0, 0].plot(hist_mask_normalized, label='Img 1 Full Mask')
axs[0, 0].plot(hist_mask_normalized_secondimg, label='Img 2 Full Mask')
axs[0, 0].set_title('Comparison: Img 1 Mask 1 vs Img 2 Mask 1')
axs[0, 0].set_xlabel('Pixel Value')
axs[0, 0].set_ylabel('Percentage')
axs[0, 0].legend()

# Comparison 2: Image 1 Mask 1 with Image 2 Mask 2
axs[0, 1].plot(hist_mask_normalized, label='Img 1 Full Mask')
axs[0, 1].plot(hist_mask2_normalized_secondimg, label='Img 2 Top Half Mask')
axs[0, 1].set_title('Comparison: Img 1 Mask 1 vs Img 2 Mask 2')
axs[0, 1].set_xlabel('Pixel Value')
axs[0, 1].set_ylabel('Percentage')
axs[0, 1].legend()

# Comparison 3: Image 1 Mask 2 with Image 2 Mask 1
axs[1, 0].plot(hist_mask2_normalized, label='Img 1 Top Half Mask')
axs[1, 0].plot(hist_mask_normalized_secondimg, label='Img 2 Full Mask')
axs[1, 0].set_title('Comparison: Img 1 Mask 2 vs Img 2 Mask 1')
axs[1, 0].set_xlabel('Pixel Value')
axs[1, 0].set_ylabel('Percentage')
axs[1, 0].legend()

# Comparison 4: Image 1 Mask 2 with Image 2 Mask 2
axs[1, 1].plot(hist_mask2_normalized, label='Img 1 Top Half Mask')
axs[1, 1].plot(hist_mask2_normalized_secondimg, label='Img 2 Top Half Mask')
axs[1, 1].set_title('Comparison: Img 1 Mask 2 vs Img 2 Mask 2')
axs[1, 1].set_xlabel('Pixel Value')
axs[1, 1].set_ylabel('Percentage')
axs[1, 1].legend()

plt.tight_layout()
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