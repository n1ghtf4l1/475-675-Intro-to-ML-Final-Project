# Reconstruction of 3D Cellular Behavior (475-675-Intro-to-ML-Final-Project)
Repository for Final Project for EN.601.475/675: Introduction to Machine Learning

## Dataset
Please find the datasets used for this project on :
We used variety of CD4+ T cell subsets ('Th0', 'Th1', 'Th2', 'Treg', 'nTh17') to train both the 2d3drecons and 3dattention model, which is stored in the training -> cropped for the brightfield images with size (32, 64, 64) and corresponding segmented images with size (32, 64, 64) in training -> segmented. Here, only Th0 subset data is shared due to limited data storage capacity, but all data can be shared upon request.

Also, we are sharing prediction outcomes from both 2d3drecons and 3dattention models. Within the 'prediction' folder, the testing dataset is within the 'brightfield' folder, prediction result is within the 'pred' folder and ground truth is within the 'truth' folder.
Outline of the dataset folder structure is as follows:

![image](https://github.com/user-attachments/assets/30f11c7c-95ee-49e2-9103-a3dbea2b969c)


## Usage
1. Preparation of the training dataset:
  1-1. prepare_training.py
     Pre-process the brightfield images and perform 3D segmentation from the fluorescent images. Saves the images to training -> cropped and training -> segmented in npy format.
  1-2. reshape_truth.py
     Reshapes the training -> segmented npy files and saves to prediction ->
