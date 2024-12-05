# Reconstruction of 3D Cellular Behavior (475-675-Intro-to-ML-Final-Project)
Repository for Final Project for EN.601.475/675: Introduction to Machine Learning

Team: Chanhong Min(cmin11), Hyunji Park(hpark111), Anubhav De(ade11), Emily Guan(eguan3)

## Dataset
Please find the datasets and trained model used for this project on: https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/cmin11_jh_edu/EgsCBi2mIv1Eo15Imr3LalsB1qZAdiR3wEuh3tN1-NkUpg?e=9VGAhk

We used variety of CD4+ T cell subsets ('Th0', 'Th1', 'Th2', 'Treg', 'nTh17') to train both the 2d3drecons and 3dattention model, which is stored in the training -> cropped for the brightfield images with size (32, 64, 64) and corresponding segmented images with size (32, 64, 64) in training -> segmented. Here, only Th0 subset data is shared due to limited data storage capacity, but all data can be shared upon request.

Also, we are sharing prediction outcomes from both 2d3drecons and 3dattention models. Within the 'prediction' folder, the testing dataset is within the 'brightfield' folder, prediction result is within the 'pred' folder and ground truth is within the 'truth' folder.
Outline of the dataset folder structure is as follows:

![image](https://github.com/user-attachments/assets/30f11c7c-95ee-49e2-9103-a3dbea2b969c)


## Usage
1. Preparation of the training dataset:
- prepare_training.py: Pre-process the brightfield images and perform 3D segmentation from the fluorescent images. Saves the images to training -> cropped and training -> segmented in npy format.
- reshape_truth.py: Reshapes the training -> segmented npy files and saves to prediction -> truth in npy format.

2. Training the model:
- train_2d3drecons.py: Train the 2D_3D ReconstructionNet
- train_3dattention.py: Train the 3D AttentionNet

3. Prediction:
- predict_2d3drecons.py: Inference step by 2D_3D ReconstructionNet
- predict_3dattention.py: Inference step by 3D AttentionNet

4. Tracking:
- tracking.py: Perform tracking using the predicted outcome of 3D AttentionNet

5. Quantification:
- extract_feature.py: Calculated motility features from the 3D trajectory and save as motility_features.csv file
- results.py: Reconstructs the 3D trajectory and learn latent features. Also generates graphs to show the figures created in powerpoint.

for running the code, path should be changed so that it fits to users' path. Step1 and Step2 can be skipped and start from step 3 when using trained model we provided. You can find the trained model in the 'saved_model' folder in the link provided very top.
