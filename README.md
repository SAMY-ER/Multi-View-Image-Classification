# Multi-View Image Classification

## Introduction

This project solves a multi-view image classification problem using two different approaches : 

1. White-box feature extractors (e.g. SIFT) and clustering for image quantization, combined with a classical machine learning algorithm (e.g. SVM) for prediction.
2. A neural network architecture (MVCNN) that inherently deals with the multi-view aspect by taking multiple images at once as an input and combining their feature maps down the road before classifying.

There is also a [Medium article](https://towardsdatascience.com/multi-view-image-classification-427c69720f30) that goes into details about the problem and these two approaches.

This project is implemented in **Python** and makes use of **Scikit-Learn** and **PyTorch** for model building and training and **OpenCV** and **Pillow** for image processing.

## Project Structure

`models` folder that contains the trained models. Specifically, kmeans and logistic regression for the first approach, and two MVCNN's (for feature extraction and fine-tuning).

`notebooks` folder that contains the jupyter notebook of this project as well as its html export for easy reading.

`resources`  folder that contains extra components required for testing the model (e.g. normalization constants and the vocabulary features).

`dataset.py` script of the PyTorch custom dataset class that reads all the view images of each data sample and returns a tensor of shape *(Views, Channels, Height, Width)*.

`network.py` script of the Multi-View Convolutional Neural Network (MVCNN) class that takes inputs of shape *(Samples, Views, Channels, Height, Width)* and returns the logits of the 6 classes. 

`trainer.py` script of a utility function to train the model while keeping track of some metrics of interest.

## Data

The data used in this problem consists of images of 833 car plugs. Each car plug is an item, and has 8 images, 6 of which correspond to orthographic projections while the other 2 are random isometric projections of the car plug. The colors found in the images are meaningless. Finally, each car plug has a unique codename and maps to a single label among a set of 6 predefined classes. This information is contained in a `train.csv` file. It is worth noting that the classes are imbalanced.

Below is an example of 8 raw images of a car plug.

![Multi-view Car Plug](/images/multi_view_car_plug.png)

Unfortunately, I can not include the data as it was privately provided by a company during a competition. However, in order to follow along certain parts of the notebook, you may need to know the structure of the data folder, as detailed below.

    Multi-View-Image-Classification/
    ├── data/
    │   ├── raw/
    │   │   ├── codename1_x1.png
    │   │   ├── codename1_x2.png
    │   │
    │   │   ...
    │   │
    │   │   ├── codename1_x8.png
    │   │
    │   │   ...
    │   │   
    │   │   ├── codename833_x8.png
    │   │   └── train.csv
    ├── models
    ├── notebooks
    ├── resources

## Results

The performance of the models is measured by a weighted accuracy to account for the class imbalance problem.


Approach 1 (SURF + LR) | Approach 2 (MVCNN)
---------------------- | ------------------
<center>92%</center> | <center>98%</center>
