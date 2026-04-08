# Sperm Morphology Classification Pipeline

## Overview
This project implements a computer vision pipeline to detect and classify sperm morphology from microscopy images. The goal is to identify bent versus normal sperm heads to support downstream tracking and analysis.

## Motivation
Accurate sperm tracking requires reliable identification of morphology. Variations such as bent sperm heads can impact movement analysis, making automated classification an important preprocessing step.

## Pipeline Steps
1. Image preprocessing (grayscale conversion, Gaussian smoothing)
2. Sperm head detection using Laplacian of Gaussian (LoG)
3. Region of Interest (ROI) extraction around detected heads
4. Segmentation and mask refinement
5. Skeletonization of the sperm structure
6. Bend angle calculation
7. Classification (bent vs. normal based on angle threshold)

## Technologies Used
- Python
- NumPy
- scikit-image
- pandas
- matplotlib

## Results
The pipeline successfully detects sperm heads and extracts structural features for classification. Improvements were made to reduce noise and limit unwanted branching in skeletonization.

## Challenges
- Noise in segmentation masks
- Branching artifacts when sperm tails exit the image frame
- Sensitivity to ROI size and thresholding parameters

## Future Improvements
- Improve segmentation robustness near image edges
- Refine skeleton pruning to eliminate false branches
- Optimize bend angle calculation for higher accuracy

## Author
Kendall Laberge
