# AI-Powered Indoor Obstacle Avoidance for UAVs

## Project Overview
This repository contains an AI perception and control loop designed to navigate indoor spaces while detecting and avoiding obstacles. Built for UAVs (drones), the system uses a Convolutional Neural Network (ResNet) to process front-facing camera feeds and issues real-time control heuristics to avoid collisions.

## Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV (`opencv-python`)
* **Deep Learning:** TensorFlow / Keras (ResNet Architecture)
* **Simulation Environment:** Microsoft AirSim (Unreal Engine)

## Architecture
1. **Perception Module (`CNNModel.py` & `ResNet_UAV.ipynb`):** Processes image data to detect spatial boundaries and obstacles.
2. **Control Loop (`CNNController.py` & `Controller.py`):** Translates AI perception outputs into velocity and yaw commands to safely navigate around detected obstacles.

## Learning Outcomes Achieved
* Implementation of Deep Learning for real-time obstacle detection.
* Integration of basic path planning and control heuristics.
* Handling sim-to-real considerations by architecting the code to interface with a virtual physics environment (AirSim) prior to hardware deployment.
