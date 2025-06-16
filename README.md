# 🧠 CIFAR-10 Image Classification using CNN

This project demonstrates image classification on the CIFAR-10 dataset using Convolutional Neural Networks (CNNs) built with Pytorch. The goal is to accurately classify images into 10 categories such as airplane, car, bird, cat, and more.

---

## 📂 Dataset Overview

The **CIFAR-10 dataset** consists of 60,000 color images in 10 classes, with 6,000 images per class:
- **Image Size**: 32x32 pixels
- **Channels**: RGB (3 channels)
- **Train/Test Split**: 50,000 training images and 10,000 test images

### 🏷️ Classes

['airplane', 'automobile', 'bird', 'cat', 'deer',
'dog', 'frog', 'horse', 'ship', 'truck']

<img src = "https://github.com/Vinit-joshi2/Image-Classification-using-CNN/blob/main/CIFAR10.png">



---

## 🚀 Project Pipeline

### 1. Exploring the CIFAR10 Dataset
- Loaded CIFAR-10 dataset from
  
```
import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split

# Dowload the dataset
dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
download_url(dataset_url, '.')

```
- Data Preprocessing
  <h4>
    Let's look inside a couple of folders, one from the training set and another from the test set.
  </h4>
  
```
airplane_files = os.listdir(data_dir + "/train/airplane")
print('No. of training examples for airplanes:', len(airplane_files))
print(airplane_files[:5])
```
<img src = "https://github.com/Vinit-joshi2/Image-Classification-using-CNN/blob/main/image1.1.png">

```
ship_test_files = os.listdir(data_dir + "/test/ship")
print("No. of test examples for ship:", len(ship_test_files))
print(ship_test_files[:5])

```
<img src = "https://github.com/Vinit-joshi2/Image-Classification-using-CNN/blob/main/image1.2.png">

  

### 2. Model Architecture (CNN)
