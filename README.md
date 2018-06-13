# **Behavioral Cloning** 
---
Implementing end to end learning for self driving cars using simulator based on Nvidia Paper : (https://arxiv.org/pdf/1604.07316v1.pdf) 

[![Introduction video](https://img.youtube.com/vi/kb53G_J8Qds/0.jpg)](https://www.youtube.com/watch?v=kb53G_J8Qds)

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[dist]: documentation/distribution.png "Model Visualization"
[train_dist1]: documentation/train_dist/plot1.png "Plot1"
[train_dist2]: documentation/train_dist/plot2.png "Plot2"
[train_dist3]: documentation/train_dist/plot3.png "Plot3"
[train_dist4]: documentation/train_dist/plot4.png "Plot4"
[val_dist1]: documentation/validation_dist/plot1.png "Plot1"
[val_dist2]: documentation/validation_dist/plot2.png "Plot2"
[val_dist3]: documentation/validation_dist/plot3.png "Plot3"
[val_dist4]: documentation/validation_dist/plot4.png "Plot4"
[drawing_arch]: documentation/drawings/Drawing1.png "Architecture"
[drawing_training]: documentation/drawings/Drawing2.png "Training"
[drawing_test]: documentation/validation_dist/Drawing3.png "Test drive"
[drawing_augmentation]: documentation/drawings/Drawing4.png "Augmentation"
[drawing_model]: documentation/drawings/Drawing5.png "MOdel"


---
## Project explanation

---
### Files & Requirements

#### Files

My project includes the following files:
* ```model.py``` containing the script to create and train the model
* ```drive.py``` for driving the car in autonomous mode
* ```model.h5``` containing a trained convolution neural network 
* ```utils.py``` containing helper functions for augmentation and generator function for lazy augmentation on batch generation
* ```README.md``` for summarizing the results
* ```videos/*``` - A videos recording of vehicle driving autonomously for ne lap around the track.

#### Requirements

##### Dependencies 

* python==3.5.2
* numpy
* matplotlib
* opencv3
* eventlet
* flask-socketio


##### pip : 
* Tensorflow GPU : https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl
* keras==2.0.6

### Running the code

#### Functionality 
For running this project you will need to install udacity simulation which can be found on [self-driving-car-sim]( https://github.com/udacity/self-driving-car-sim) git repository.


#### Running
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### Code Readability
The code in model.py uses a Python generator located in utils.py, to generate data for training rather than storing the training data in RAM. The model.py code is clearly organized and comments are included where needed.


The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
