# Modified MNIST Challenge 

This project is a collaboration with Carl Perreault-Lafleur and Khoren Ponsin.

The goal of this project is to implement a model that predicts the highest value digit in an image containing three
randomly placed, randomly rotated MNIST digits in a gray-scale image. The models were used to participate in a Kaggle
Competition titled “Modified MNIST". The goal of this competition was to achieve the top prediction accuracy score on
a provided test dataset. Through a variety experiments used to tune hyperparameters and determine the top performing
neural network architecture, we arrived at a model which achieved a test accuracy score of 96.033%. 

Our results are summarized in the following:
https://drive.google.com/file/d/1JDnIOXBIhBpj9LOTw9BVav3bUnkK8g8u/view?usp=sharing

- The notebook 'Digit_Split_Model' contains all the results for the Intuitive Approach part of the project.
- The notebook 'Double_Data' contains the code used to run the 4 layers CNN on the doubled dataset.
- The creation of the additional images can be done using the following ressources: https://github.com/shaohua0116/MultiDigitMNIST
- The notebooks 'Modified_VGG-16vsVGG-16' and 'Project_3_ml_vgg16' contain all the results for the VGG-16 part of the project.
- The notebooks 'Project_3_ml_tuning_lr_0_0001', and 'Project_3_ml_tuning_lr_0_00005' contain all the results for the Tuning the Learning Rate part of the project.
- The notebooks 'Batch_Size_Tuning' contain all the results for the Tuning the Batch Size part of the project.

In those notebooks, we used the following packages:

- import pandas as pd
- %matplotlib inline
- from matplotlib import pyplot as plt
- import numpy as np
- from PIL import Image
- import tensorflow as tf
- from keras.models import Sequential
- from keras.layers import Dense, Conv2D, Flatten
- import time
- import datetime as dt
- import progressbar
- import pickle as pkl
- from skimage.transform import rotate
- import skimage
- import os
- from os import listdir
- import chainer
- import torch
- from numpy.random import randint
- from skimage.measure import label, regionprops
- from __future__ import print_function
- import argparse
- import torch.nn as nn
- from skimage.transform import resize
- import torch.nn.init as init
- import torch.nn.functional as F
- from torch.autograd import Variable
- import torch.optim as optim
- from torch.autograd import Variable
- from sklearn.model_selection import train_test_split
- from sklearn.metrics import accuracy_score, classification_report
- import torch.utils.data as data_utils
