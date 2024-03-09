import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from cv2.typing import MatLike
from typing import Tuple, Dict

type Histogram = list[list[float]]
# (x, y, w, h) - (x,y) is the top left point
type BoundingBox = tuple[int, int, int, int]

bboxes_person_img1: dict[BoundingBox, str] = {
    (232,128,70,269): 'girl in black vest 1',
    (333,136,85,219): 'girl in brown vest 1',
    (375,271,156,188): 'blue hat man',
    (321,269,123,189): 'girl in black vest 2',
    (463,251,112,206): 'girl in brown vest 2',
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
    correlations_file = open('correlations.txt', 'a')
    results_file = open('results.txt', 'a')
    best_results: Dict[tuple[int,int,int,int], 
                       Dict[
                           str, 
                           float
                        ]
                    ] = {}

    for ref_bbox, ref_d in reference_histograms.items():
        for ref_mask, ref_hist in ref_d.items():
            for img_name, dataset_d in dataset_histograms.items():
                correlations_file.write('==> Comparing to IMG ' + img_name + '.png\n\n')
                for dataset_bbox, dataset_d2 in dataset_d.items():
                    for dataset_mask, dataset_hist in dataset_d2.items():
                        correlation = calculate_correlation(ref_hist, dataset_hist)
                        if not ref_bbox in best_results:
                            best_results[ref_bbox] = [correlation]
                        else:
                            if len(best_results.get(ref_bbox)) < 100:
                                best_results.get(ref_bbox).append(correlation)
                            else:
                                if min(best_results.get(ref_bbox)) < correlation:
                                    best_results.get(ref_bbox).remove(min(best_results.get(ref_bbox)))
                                    best_results.get(ref_bbox).append(correlation)

                            # if not ref_bbox in best_results:
                        #     best_results[ref_bbox] = [(img_name, correlation)]
                        # else:
                        #     if len(best_results.get(ref_bbox)) < 100:
                        #         best_results.get(ref_bbox).append((img_name, correlation))
                        #     else:
                        #         if min(best_results.get(ref_bbox)) < correlation:
                        #             best_results.get(ref_bbox).remove(min(best_results.get(ref_bbox)))
                        #             best_results.get(ref_bbox).append(correlation)

                        correlations_file.write(
                            f"Comparing ref_hist from ref_bbox/ref_mask: {ref_bbox}/{ref_mask} with ds_hist from ds_bbox/ds_mask:{dataset_bbox}/{dataset_mask} => {correlation}"  + "\n"
                        )
    for k,v in best_results.items():
        results_file.write(str(k) + ':' + str(v) + '\n')

def main():
    ref_histograms = create_reference_histograms()
    dataset_histograms = compose_dataset_histograms('labels.txt')
    compare_reference_dataset(ref_histograms, dataset_histograms)

if __name__ == '__main__':
    main()