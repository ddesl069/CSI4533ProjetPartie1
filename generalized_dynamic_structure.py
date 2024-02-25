import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import itertools

def calculate_histogram(img, bbox, mask_type):
    mask = np.zeros(img.shape[:2], np.uint8)
    x, y, w, h = bbox
    if mask_type == 'full':
        mask[y:y+h, x:x+w] = 255
    elif mask_type == 'half':
        mask[y:y+h//2, x:x+w] = 255
    hist = cv.calcHist([img], [0], mask, [256], [0, 256])
    normalized_hist = (hist / hist.sum()) * 100
    return normalized_hist

def calculate_correlation(hist1, hist2):
    return cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)


img1 = cv.imread('./image_test/1636738315284889400.png', cv.IMREAD_GRAYSCALE)
assert img1 is not None, "file could not be read, check with os.path.exists()"

img2 = cv.imread('./image_test/1636738357390407600.png', cv.IMREAD_GRAYSCALE)
assert img2 is not None, "file could not be read, check with os.path.exists()"

bboxes_img1 = [(232,128,70,269)]  # Add more bounding boxes if needed
bboxes_img2 = [(463,251,112,206),(321,269,123,189)]  #Add more bounding boxes if needed

# Generate histograms for all bbox-mask combinations
histograms_img1 = {bbox_i: {mask: calculate_histogram(img1, bbox, mask) 
                            for mask in ['full', 'half']} 
                   for bbox_i, bbox in enumerate(bboxes_img1)}

histograms_img2 = {bbox_i: {mask: calculate_histogram(img2, bbox, mask) 
                            for mask in ['full', 'half']} 
                   for bbox_i, bbox in enumerate(bboxes_img2)}

dataset_histograms = {}
with open('labels.txt') as f:
    labels = f.readlines()
    for line in labels:
        img_name, x, y, w, h = line.split(',')
        x, y, w, h = int(x), int(y), int(w), int(h)

        img = cv.imread(f'./dataset/{img_name}.png', cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"{img_name}.png could not be read, check with os.path.exists()")
            continue

        if not img_name in dataset_histograms:
            dataset_histograms[img_name] = [calculate_histogram(img, (x,y,w,h), mask) for mask in ['full', 'half']]
        else:
            print(dataset_histograms.get(img_name))
            dataset_histograms.update({img_name: dataset_histograms.get(img_name) + [calculate_histogram(img, (x,y,w,h), mask) for mask in ['full', 'half']]})

#then compare

# Compare all combinations
for (i1, masks1), (i2, masks2) in itertools.product(histograms_img1.items(), histograms_img2.items()):
    for mask1, hist1 in masks1.items():
        for mask2, hist2 in masks2.items():
            correlation = calculate_correlation(hist1, hist2)
            print(f"Comparison ({i1}, {i2}, '{mask1}-{mask2}') has correlation: {correlation:.4f}")


