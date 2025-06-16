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

- Image size

  ```
  img , label = dataset[0]
  print(img.shape , label)

  ```

<img src = "https://github.com/Vinit-joshi2/Image-Classification-using-CNN/blob/main/image1.3.png">

- View image

  ```
  import matplotlib.pyplot as plt
  %matplotlib inline
  
  def show_example(img , label):
    print("Label :" , dataset.classes[label]  , "(" + str(label)+ ")")
    plt.imshow(img.permute(1 , 2 , 0))

  show_example(*dataset[344])
  
  ```

<img src = "https://github.com/Vinit-joshi2/Image-Classification-using-CNN/blob/main/image1.4.png">

  

### 2. Training and Validation Datasets

```
val_size = 5000
train_size = len(dataset) - val_size

train_ds , val_ds = random_split(dataset , [train_size , val_size])
len(train_ds)  , len(val_ds)
```
<h4> 
We can now create data loaders for training and validation, to load the data in batches
</h4>

```
from torch.utils.data.dataloader import DataLoader
batch_size = 128

train_dl = DataLoader(train_ds , batch_size , shuffle = True , num_workers=4 , pin_memory=True)
val_dl = DataLoader(val_ds , batch_size*2 , num_workers=4 , pin_memory=True)

```

<h4>
  We can look at batches of images from the dataset using the make_grid method from torchvision. Each time the following code is run, we get a different bach, since the sampler shuffles the indices before creating batches.
</h4>

```
from torchvision.utils import make_grid

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break

show_batch(train_dl)
```

<img src = "https://github.com/Vinit-joshi2/Image-Classification-using-CNN/blob/main/image2.1.png">

```
show_batch(val_dl)
```
<img src = "https://github.com/Vinit-joshi2/Image-Classification-using-CNN/blob/main/image2.2.png">

