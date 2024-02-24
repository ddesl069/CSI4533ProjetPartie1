# Detection of People from Frame to Frame

## Goal

Detect people on video camera feed from frame to frame.

## Requirements

1. Two Histograms per person
    - Take histogram of (whatever we can see of) the person
    - Take another histogram of the upper half of the first histogram
2. Compararaison
    - Compare an individual's histograms with the histograms of the others 
3. Results:
    - Verify and take notes of accuracy

## Logic

### With the two test images

1 - Draw masks
2 - Make the 2 histograms from them
3 - Compare one person's histograms with other people's histograms to find correlation %

### With all images

1 - Draw masks of test images
2 - Make histograms of the 5 people on the test images
3 - Load an image
4 - Compare each of the 5 people of test images to everyone on the image
5 - Repeat 3,4 until done -- Should be 500 results
6 - Pick 100 best results
7 - From those 100 best results, manually verify accuracy

## Questions
- Why do we have to normalize histograms?
    - Because some histograms' number of pixels with vary a lot with other, skewing the results without normalizing.

## Additional notes
- Can use GrabCut to get rid of background if wanted