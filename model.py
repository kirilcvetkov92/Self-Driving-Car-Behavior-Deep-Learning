import csv
import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers import Dropout
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from utils import preprocess
import matplotlib.pyplot as plt
from random import shuffle
lines = []
import random
from utils import add_salt_pepper_noise


images = []
measurements = []

import os
import csv

samples = []
distr=[]
with open('nov_trening/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        distr.append(float(line[3]))
# with open('first_lane_slow/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         samples.append(line)
#         distr.append(float(line[3]))


with open('2_side_lane/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        distr.append(float(line[3]))

# with open('train_second_shadow/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         samples.append(line)
#         distr.append(float(line[3]))


# with open('2_lane_one_more/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         samples.append(line)
#         distr.append(float(line[3]))

# with open('full_speed/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         samples.append(line)
#         distr.append(float(line[3]))

with open('lane2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        distr.append(float(line[3]))

# with open('2_side_lane/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         samples.append(line)
#         distr.append(float(line[3]))




# with open('repaired_data/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         samples.append(line)
#         distr.append(float(line[3]))
# #
# with open('lane_2_new/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         samples.append(line)
#         distr.append(float(line[3]))
# #
# with open('repaired_2/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         samples.append(line)
#         distr.append(float(line[3]))
#
# with open('repaired3/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         samples.append(line)
#         distr.append(float(line[3]))
#
# with open('repaired4/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         samples.append(line)
#         distr.append(float(line[3]))

# with open('second_lane/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         samples.append(line)
#         distr.append(float(line[3]))


# with open('sides/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         samples.append(line)
#         distr.append(float(line[3]))

n, bins, patches = plt.hist(np.array(distr), bins=20)
plt.show()


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn



def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = 320 * np.random.rand(), 0
    x2, y2 = 320 * np.random.rand(), 160
    xm, ym = np.mgrid[0:160, 0:320]


    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def generator(samples, batch_size=32, is_validation=False):
    num_samples = len(samples)
    range_x = 100
    range_y = 10
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                corrections = [0,0.2,-0.2]

                for i in range(3):
                    source_path = batch_sample[i]
                    split = source_path.split('\\')
                    filename = split[-1].lstrip()
                    file = split[-3]
                    #print(split)
                    #print(filename)
                    current_path = file+'/IMG/'+ filename
                    #print(current_path)
                    image = cv2.imread(current_path)


                   # print (source_path)
                    images.append(preprocess(image))
                   # print(image)
                    measurement = float(batch_sample[3]) + corrections[i]
                    angles.append(measurement)

                    images.append(preprocess(cv2.flip(image, 1)))
                    #measurement*=-1.0
                    angles.append(measurement*-1.0)

                    if not is_validation:
                        if random.random() < 0.5:
                            image = cv2.flip(image, 1)
                            measurement *= -1.0

                        image = add_salt_pepper_noise(image)
                        # t = plt.imshow(image)
                        # plt.show()
                        image, measurement = random_translate(image, measurement, range_x, range_y)

                        image = random_shadow(image)
                        image = random_brightness(image)

                        images.append(preprocess(image))
                        angles.append(measurement)

            #print("\n")
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #
            # n, bins, patches = plt.hist(np.array(y_train), bins=20)
            # plt.show()
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32, is_validation=True)


model = Sequential()
#model.add(Lambda(lambda x : x/float(255) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Conv2D(24,5,5, subsample=(2,2), border_mode='valid',  activation="elu"))
#model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(36,5,5, subsample=(2,2), activation="elu"))
#model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(48,5,5, subsample=(2,2),activation="elu"))
#model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(64,3,3, activation="elu"))
#model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(64,3,3,activation="elu"))
model.add(BatchNormalization())

#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(50, activation='elu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

#mae
model.compile(loss='mse', optimizer='adam')


checkpoint = ModelCheckpoint('checkpoints/ks-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=False,
                                 mode='auto')


#model = load_model('checkpoints/rs-008.h5')

history_object = model.fit_generator(train_generator, len(train_samples)//32, validation_data=validation_generator,nb_val_samples=len(validation_samples)//100, nb_epoch=250, callbacks=[checkpoint])
model.save('model11'
           '.h5')


### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()