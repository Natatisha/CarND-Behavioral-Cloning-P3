# **Behavioral Cloning Project** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[loss]: ./writeup_images/loss.png "Loss"
[center]: ./writeup_images/center.jpg "Center"
[center1]: ./writeup_images/center_1.jpg "Center"
[right]: ./writeup_images/right.jpg "Right"
[right1]: ./writeup_images/right_1.jpg "Right"
[left]: ./writeup_images/left.jpg "Left"
[left1]: ./writeup_images/left_1.jpg "Left"
[cropped]: ./writeup_images/cropped.png "Cropped"
[cropped1]: ./writeup_images/cropped_1.png "Cropped"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `README.md` summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with the following architecture: 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320 RGB image   							| 
| Normalization layer | Normalizes input and maps to zero mean |
| Cropping layer | Performs image cropping using `Cropping2D` class to size of 65x320 |  
| Convolution 5x5     	| Outputs 61x316x24, RELU activation 	|
| Convolution 5x5     	| Outputs 57x312x36, RELU activation	|
| Max pooling	      	| Outputs 28x156x36 				|
| Convolution 5x5     	| Outputs 24x152x48, RELU activation		|
| Convolution 3x3     	| Outputs 22x150x64, RELU activation	 	|
| Max pooling	      	| Outputs 11x75x64 			|
| Convolution 3x3     	| Outputs 9x73x64, RELU activation	 	|
| Flatten     |  42048      |
| Dense		|     100 |
| Dense		|     50 									|
| Output		|     1									|
| *Trainable params:* | *4,341,349* |

All code related to building neural network is located in `build_network` function in `model.py` file. 

#### 2. Attempts to reduce overfitting in the model
 
 To make sure that the model doesn't overfit I've used different data for training and validation. After each epoch, I've compared training and validation losses and made sure that validation loss is not much higher than training loss. 
 Also, each convolution layer contains L2 regularization with beta=0.01. This helped to avoid overfitting, and after all validation loss after each epoch was even smaller than the training loss.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (`model.py` line 84).
Other hyperparameters to tune: 
- batch size: I've chosen 32, it gave me solid performance and good results
- number of epochs: the maximum number of epochs is 8, but I also used early stopping technique to terminate learning if validation loss isn't improving (reducing) during 2 last epochs (`model.py` line 117).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The primary strategy for deriving a model architecture was to create a small and simple CNN (like LeNet) at first, test its performance and verify that the model is learning at all. This approach helps to find mistakes and bugs at the early stages when the code is less complicated. 
After that, I've made a deeper network and collected more data. The final network architecture was described above. The main idea was taken from [this](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) article. I've added Max Pooling layers to reduce the number of parameters and fasten learning. 
To deal with the overfitting, I added L2 regularization. I also tried adding Batch Normalization to convolutional layers, but the network had the worse overall performance and (surprisingly) it took longer for the model to converge.
To make the learning process easier, I've also added two callbacks: 
1. Early stopping to avoid wasting time for redundant training epochs.
2. Model checkpoint to save the model with the best weights (the one with the lowest validation loss).

The final step was to run the simulator to see how well the car was driving around track one. I've modified the `drive.py` script to increase the speed to 23 m/h, to make things more interesting. There were a few spots where the vehicle fell off the track, mostly the sharp turns. To improve the driving behavior in these cases, I collected more data on both tracks and trained the model again. 

Finally, the vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My final model architecture consists of (besides pre-processing layers): 
 - two convolution layers (5x5, 5x5)
 - pooling layer 
 - two convolution layers (5x5, 3x3)
 - pooling layer 
 - convolution layer (3x3)
 - flatten layer
 - dense layer (100)
 - dense layer (50)
 - output layer (1)

#### 3. Creation of the Training Set & Training Process

At the beginning of the training, I've used only provided data. When the network became deeper, I've realized that it's not enough, and collected more data by recording one more full lap on the first track. 

I've also noticed that the vehicle falls during left turns and recorded more driving samples of the left turn (I've used the second track for it because it has more turns and also I believed this would help the model to generalize better).

To get even more data, I've also used the images from the left and the right cameras (and added the correction angle of &#177;0.24). Check out some of the sample images from the first and the second track below: 

![alt text][left] ![alt text][left1] 

![alt text][center] ![alt text][center1]

![alt text][right] ![alt text][right1]

After the collection process, I had 40760 number of data points. 

I then preprocessed this data by dividing pixel values by 255, so now we have pixel values in the range of [0, 1]. To make all values have zero mean I've subtracted 0.5, as it was suggested in the lesson. I've also cropped images to remove unnecessary information and reduce dimensionality. The cropped images look like this: 

![alt text][cropped]

![alt text][cropped1]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 
I used this training data for training the model. The validation set helped determine if the model was over or underfitting. I've used early stopping, but eventually it wasn't needed, because the model was training all 8 episodes. The lowest validation was achieved on the epoch 6 (so we saved the model for the further use), as we can see on the plot below: 

![alt text][loss]

As we can see, the loss in constantly decreasing, which means that the model is learning. Also, the validation loss is usually lower that training loss, so regularization worked and the model doesn't overfit. 

### Summary 

The final results are recorded on the [video.mp4](video.mp4). The vehicle can pass the first track with a speed of 23 m/hour and stay in line, which is not bad results.
However, there is still room for improvement: the vehicle could move more smoothly, and we can increase the test drive speed. Also, we can train the model for the second track, which is more harder. 
To achieve this, we'll probably need: 
 - much mode data from both tracks, especially the second one
 - more complicated images pre-processing 
 - data augmentation 
 - we can use transfer learning for even better results 
 
 
 
