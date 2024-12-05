[dataset_structure.txt](https://github.com/user-attachments/files/18028777/dataset_structure.txt)# Reconstruction of 3D Cellular Behavior (475-675-Intro-to-ML-Final-Project)
Repository for Final Project for EN.601.475/675: Introduction to Machine Learning

## Dataset
Please find the datasets used for this project on :
We used variety of CD4+ T cell subsets ('Th0', 'Th1', 'Th2', 'Treg', 'nTh17') to train both the 2d3drecons and 3dattention model, which is stored in the training -> cropped for the brightfield images with size (32, 64, 64) and corresponding segmented images with size (32, 64, 64). Here, only Th0 subset data is shared due to limited data storage capacity, but all data can be shared upon request.

Also, we are sharing prediction outcomes from both 2d3drecons and 3dattention models. The testing dataset is within the 'brightfield' folder, prediction result is within the 'pred' folder and ground truth is within the 'truth' folder.
Outline of the dataset folder structure is as follows:

![image](https://github.com/user-attachments/assets/30f11c7c-95ee-49e2-9103-a3dbea2b969c)
[Uploading dataset_dataset
└───training
│   └───Th0
│       │   Th0_3D_REP1_Merging_001_t00_z00_RAW_ch00.tif
│       │   Th0_3D_REP1_Merging_001_t00_z00_RAW_ch01.tif
│       │   ...
│       └───cropped
│       │   img_t000.tif
│       │   img_t001.tif
│       │   ...
│       └───segmented
│       │   segmented_t000.tif
│       │   segmented_t001.tif
│       │   ...
└───prediction
│   └───2d3drecons
│   │   └───Th0
│   │       └───brightfield
│   │       │   bf_t000.npy
│   │       │   bf_t001.npy
│   │       │   ...
│   │       └───pred
│   │       │   pred_t000.npy
│   │       │   pred_t001.npy
│   │       │   ...
│   │       └───truth
│   │       │   truth_t000.npy
│   │       │   truth_t001.npy
│   │       │   ...      
│   └───3dattention
│       └───Th0
│           └───brightfield
│           │   bf_t000.npy
│           │   bf_t001.npy
│           │   ...
│           └───pred
│           │   pred_t000.npy
│           │   pred_t001.npy
│           │   ...
│           └───truth
│           │   truth_t000.npy
│           │   truth_t001.npy
│           │   ...      structure.txt…]()

