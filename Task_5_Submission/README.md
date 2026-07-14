# Task 5 - AI Self-Driving Cars: Autonomous Stack

## Overview
This repository contains a lightweight, integrated autonomous driving stack implemented in pure Python. The project demonstrates core Advanced Driver Assistance Systems (ADAS) perception and control components, processing a dashcam video feed frame-by-frame to detect road boundaries, identify surrounding vehicles, and simulate steering corrections.

This submission fulfills the requirements for **Task 5**, combining concepts from computer vision, machine learning, and control theory into a single pipeline.

## Features & Pipeline Architecture

The main execution loop in `autonomous_stack.py` orchestrates three primary phases:

### 1. Lane Detection (Computer Vision)
* **Techniques Used:** Grayscale conversion, Gaussian Blurring, Canny Edge Detection, and Hough Transform (`cv2.HoughLinesP`).
* **Implementation:** The pipeline isolates the region of interest (the road) using a polygonal mask, detects line segments, and mathematically interpolates positive and negative slopes to draw solid boundary lines for the left and right lanes. The visual overlay is cropped to the horizon line to prevent infinite rendering.

### 2. Vehicle Perception (Machine Learning)
* **Techniques Used:** Histogram of Oriented Gradients (HOG), Color Histograms, Spatial Binning, and a Support Vector Machine (SVM).
* **Implementation:** A multi-scale sliding window approach scans the lower half of the frame. Image patches are passed through a pre-trained Linear SVM to classify "car" vs. "non-car." To eliminate false positives, bounding boxes are accumulated over a 5-frame history window, processed into a heat map, and thresholded before drawing the final bounding boxes.

### 3. Control Simulation (PID)
* **Techniques Used:** Proportional-Integral-Derivative (PID) Controller.
* **Implementation:** A custom Python class simulates steering angle calculations based on Cross-Track Error (CTE). The script simulates CTE drift and outputs the corresponding steering adjustment required to keep the vehicle centered.

## Technical Stack & Libraries
* **Language:** Python 3.x
* **Core Libraries:** `OpenCV` (cv2), `NumPy`, `scikit-learn`, `scikit-image`, `SciPy`.

## Installation & Setup

Development Notes & Modernization Fixes
Numpy Deprecation: Implemented a shim (np.int = int) to handle modern Numpy environments where legacy np.int aliases have been removed.

Scikit-Image Updates: Updated kwargs in HOG feature extraction from the legacy British spelling (visualise) to the modern American spelling (visualize).

Pickle Deserialization: Built a sys.modules translation map to allow modern scikit-learn (v1.x+) to successfully unpickle models trained on legacy versions (v0.18).

To run this project locally, you must install the required dependencies. The pre-trained SVM model was originally serialized in an older version of scikit-learn, so the master script includes compatibility shims to map legacy module paths to modern environments.

1. **Install Dependencies:**
   ```bash
   pip install numpy opencv-python scikit-learn scikit-image scipy matplotlib
