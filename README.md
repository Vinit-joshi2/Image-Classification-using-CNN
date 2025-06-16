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

### 3. Defining the Model (Convolutional Neural Network)

<h4>
  The 2D convolution is a fairly simple operation at heart: you start with a kernel, which is simply a small matrix of weights. This kernel “slides” over the 2D input data, performing an elementwise multiplication with the part of the input it is currently on, and then summing up the results into a single output pixel.
</h4>


<h4>
  The Conv2d layer transforms a 3-channel image to a 16-channel feature map, and the MaxPool2d layer halves the height and width. The feature map gets smaller as we add more layers, until we are finally left with a small feature map, which can be flattened into a vector. We can then add some fully connected layers at the end to get vector of size 10 for each image.
</h4>

<img src = "https://github.com/Vinit-joshi2/Image-Classification-using-CNN/blob/main/image3.1.png">

```
class Cifar10CnnModel(ImageClassificationBase):

  def __init__(self):
    super().__init__()

    self.network = nn.Sequential(
        #  input - 3 x 32 x 32
        nn.Conv2d(3,32, kernel_size=3 , padding  = 1) ,
        # output 32 x 32 32
        nn.ReLU(),
        # output 32 x 32 32
        nn.Conv2d(32 , 64 , kernel_size = 3 , stride = 1 , padding = 1) ,
        nn.ReLU(),
        # 64  x 32 x 32
        nn.MaxPool2d(2,2), # output - 64 x 16 x 16

        # input 64 x 16 x 16
        nn.Conv2d(64 , 128 ,kernel_size=3 , stride= 1 , padding = 1),
        nn.ReLU(),
        # output 128 x 16 x 16
        nn.Conv2d(128 , 128 , kernel_size = 3 , stride = 1 , padding = 1),
        nn.ReLU(),
        # output 128 x 16 x 16
        nn.MaxPool2d(2,2) , # output 128 x 8 x 8

        # input 128  x 8 x 8
        nn.Conv2d(128 , 256 , kernel_size=3 , stride = 1 , padding = 1),
        nn.ReLU(),
        # output  256 x 8  x 8
        nn.Conv2d(256 , 256  , kernel_size=3 , stride  = 1 , padding=1),
        nn.ReLU(),
        # output 256 x 8 x 8
        nn.MaxPool2d(2,2), # output 256 x 4 x 4

        #  convert it into 1D
        nn.Flatten(),
        nn.Linear(256 * 4 * 4 , 1024),
        nn.ReLU(),
        nn.Linear(1024 , 512),
        nn.ReLU(),
        nn.Linear(512 , 10),


  )

  def forward(self , xb):
      return self.network(xb)

```

### 4. Training the Model

<h4>We can use the exact same fit and evaluate functions</h4>

```
def evaluate(model , val_loader):
  model.eval()
  outputs = [model.validation_step(batch) for batch in val_loader]

  return model.validation_epoch_end(outputs)


def fit(epochs , lr , model , train_loader  , val_loader ,opt_func = torch.optim.SGD):

  history = []
  optimizer = opt_func(model.parameters() , lr)
  for epoch in range(epochs):
    # Training Phase
    model.train()
    train_losses = []
    for batch in train_loader:
      loss = model.training_step(batch)
      train_losses.append(loss)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()


    # Validation phase


    result = evaluate(model , val_loader)
    result["train_loss"] = torch.stack(train_losses).mean().item()
    model.epoch_end(epochs , result)
    history.append(result)

  return history

```

```
evaluate(model , val_dl)
```

<img src = "https://github.com/Vinit-joshi2/Image-Classification-using-CNN/blob/main/image4.1.png">

<h4>
  The initial accuracy is around 10%, which is what one might expect from a randomly intialized model (since it has a 1 in 10 chance of getting a label right by guessing randomly).
</h4>

```
num_epochs = 10
opt_func = torch.optim.Adam
lr = 0.001

history = fit(num_epochs , lr , model , train_dl , val_dl , opt_func)
```

<img src = "https://github.com/Vinit-joshi2/Image-Classification-using-CNN/blob/main/image4.2.png">

<h4>
  We can also plot the valdation set accuracies to study how the model improves over time.
</h4>

```
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');

plot_accuracies(history)

```

<img src = "https://github.com/Vinit-joshi2/Image-Classification-using-CNN/blob/main/image4.3.png>

<h4>
  Our model reaches an accuracy of around 75%, and by looking at the graph, it seems unlikely that the model will achieve an accuracy higher than 80% even after training for a long time. This suggests that we might need to use a more powerful model to capture the relationship between the images and the labels more accurately. This can be done by adding more convolutional layers to our model, or incrasing the no. of channels in each convolutional layer, or by using regularization techniques.
</h4>

```
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');

plot_losses(history)

```


<img src = "https://github.com/Vinit-joshi2/Image-Classification-using-CNN/blob/main/image4.3.png>

<h4>
  Initialy, both the training and validation losses seem to decrease over time. However, if you train the model for long enough, you will notice that the training loss continues to decrease, while the validation loss stops decreasing, and even starts to increase after a certain point!
</h4>
