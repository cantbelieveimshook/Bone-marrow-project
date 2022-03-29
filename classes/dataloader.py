import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw 
from numpy import asarray
from google.colab import drive
import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from math import sqrt
import sklearn
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf 
from tensorflow import keras
from keras import layers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, ReLU
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model, Sequential
import os
from os import listdir
import torch.optim as optim

class BoneMarrowDataset(Dataset):
  def __init__(self, dataset_info_path):
    self.dataset_info_path = dataset_info_path
    self.df = pd.read_csv(self.dataset_info_path)
  def __len__(self):
    return self.df.shape[0]
  def __getitem__(self, idx):
    return torch.from_numpy(plt.imread(self.df.iloc[idx, 0]).transpose(2, 0, 1).reshape(1, 3, 250, 250)/255), self.df.iloc[idx, 1]
