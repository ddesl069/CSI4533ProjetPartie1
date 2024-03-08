import numpy as np
import cv2 as cv
import itertools
from matplotlib import pyplot as plt
from cv2.typing import MatLike
from typing import Tuple, Dict

type Histogram = list[list[float]]
# (x, y, w, h) - (x,y) is the top left point
type BoundingBox = tuple[int, int, int, int]

bboxes_person_img1: dict[BoundingBox, str] = {
    (232,128,70,269): 'girl in black vest',
    (333, 128, 84, 256): 'girl in brown vest',
    (382, 271, 146, 76): 'blue hat man',
    (332, 75, 31, 37): 'man in back right',
    (258, 75, 31, 32): 'man in back left',
}

def calculate_histogram(img: MatLike, bbox: BoundingBox, mask_type: str) -> Histogram:
    mask = np.zeros(img.shape[:2], np.uint8)
    x, y, w, h = bbox

    if mask_type == 'full':
        mask[y:y+h, x:x+w] = 255
    elif mask_type == 'half':
        mask[y:y+h//2, x:x+w] = 255
        
    hist = cv.calcHist([img], [0], mask, [256], [0, 256])
    normalized_hist = (hist / hist.sum()) * 100
    
    return normalized_hist

def calculate_correlation(hist1: Histogram, hist2: Histogram) -> float:
    return cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)

def create_reference_histograms():
    """
    Makes the histograms of all (5) people that appear in the test images.
    Those histograms will then be compared to other histograms in the dataset
    to find matches.
    """
    img1 = cv.imread('./image_test/1636738315284889400.png', cv.IMREAD_GRAYSCALE)
    assert img1 is not None, "file could not be read, check with os.path.exists()"

    """
    Description of the structure of reference_histograms:
        {
            (x,y,w,h): {
                full: histogram, 
                half: histogram
            }, 
            //, 
            ...
        } ==> 2 histograms * 5 people => 10
    """
    reference_histograms: Dict[
        Tuple[int,int,int,int], 
        Dict[str, Histogram]
    ] = {
        bbox: {
            mask: calculate_histogram(img1, bbox, mask)
            for mask in ['full', 'half']
        }
        for _, bbox in enumerate(list(bboxes_person_img1.keys()))
    }
    
    return reference_histograms

def compose_dataset_histograms(labels_filepath: str):
    """Compose half and full histograms of 
    all bounding boxes for each image of the dataset.

    Parameters:
    labels_filepath: str -- path to file containing the 
                            labels for our bounding boxes
    """
    dataset_histograms: Dict[
        str: Dict[
            Tuple[int,int,int,int], 
            Dict[str, Histogram]
        ]
    ] = {}

    with open(labels_filepath) as labels:
        for line in labels.readlines():
            img_name, x, y, w, h = line.split(',')
            x, y, w, h = int(x), int(y), int(w), int(h)

            img = cv.imread(f'./dataset/{img_name}.png', cv.IMREAD_GRAYSCALE)
            if img is None:
                #print(f"{img_name}.png could not be read, check with os.path.exists()")
                continue

            if not img_name in dataset_histograms:
                dataset_histograms[img_name] = {
                    (x,y,w,h): {
                        mask: calculate_histogram(img, (x,y,w,h), mask) 
                              for mask in ['full', 'half']
                    }
                }
            else:
                dataset_histograms.get(img_name).update({
                    (x,y,w,h): {
                        mask: calculate_histogram(img, (x,y,w,h), mask) 
                              for mask in ['full', 'half']
                    }
                })

    return dataset_histograms

def compare_reference_dataset(reference_histograms, dataset_histograms):
    # reference_histograms = rh

    # dataset_histograms: Dict[
    #     str: Dict[
    #         Tuple[int,int,int,int], 
    #         Dict[str, Histogram]
    #     ]
    # ] = compose_dataset_histograms('labels.txt')
    output_file = open('output_file.txt', 'a')
    count = 0
    for ref_bbox, ref_d in reference_histograms.items():
        for ref_mask, ref_hist in ref_d.items():
            for img_name, dataset_d in dataset_histograms.items():
                print('IMG ' + img_name)
                for dataset_bbox, dataset_d2 in dataset_d.items():
                    for dataset_mask, dataset_hist in dataset_d2.items():
                        correlation = calculate_correlation(ref_hist, dataset_hist)
                        count += 1

                        s = f"Comparing ref_hist from ref_bbox/ref_mask: {ref_bbox}/{ref_mask} with ds_hist from ds_bbox/ds_mask:{dataset_bbox}/{dataset_mask} => {correlation:.4f}"
                        output_file.write(s)

    print(count)
    # for img_name, d in dataset_histograms.items():
    #     print('IMAGE ' + img_name)
    #     for bbox, d2 in d.items():
    #         print('bbox in dataset: ' + bbox)
    #         for h in d2.values():
    #             for h2 in [reference_histograms.keys()]:
                    
    #                 correlation = calculate_correlation(h, h2)
    #                 print(f"Comparison ({i1}, {i2}, '{mask1}-{mask2}') has correlation: {correlation:.4f}")()

    # for (), () in itertools.product(reference_histograms):


if __name__ == '__main__':
    ref_histograms = create_reference_histograms()
    dataset_histograms = compose_dataset_histograms('labels.txt')
    compare_reference_dataset(ref_histograms, dataset_histograms)

# print(compose_dataset_histograms('labels.txt').get("1636738303879021100"))