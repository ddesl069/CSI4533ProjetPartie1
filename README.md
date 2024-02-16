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

1 - Draw masks
2 - Make the 2 histograms from them
3 - Compare person's histograms with other people's histograms to find matches
4 - 

## Questions
- histogrammes normalisé? Why?

## Additional notes
- grabcut to get rid of background if wanted