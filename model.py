import csv
import cv2
import numpy as np

lines = []


images = []
measurements = []

with open ('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


for line in lines :
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = 'data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

lines = []

with open ('data_counter_cc/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

corrections=[0.2,0,-0.2]
for line in lines :
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = 'data_counter_cc/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])+corrections[i]
        measurements.append(measurement)
        #print(image.shape)

lines = []

with open ('curved_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

for line in lines :
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = 'curved_data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)



# lines = []
#
# with open ('very_curved/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         lines.append(line)
#
# for line in lines :
#     for i in range(3):
#         source_path = line[i]
#         filename = source_path.split('\\')[-1]
#         current_path = 'very_curved/IMG/' + filename
#         image = cv2.imread(current_path)
#         images.append(image)
#         measurement = float(line[3])
#         measurements.append(measurement)


lines = []

with open ('curved_smooth/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

for line in lines :
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = 'curved_smooth/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

# X_train = np.array(images)

# X_train = np.array(images)
# y_train = np.array(measurements)
#
#print(X_train.shape)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers import Dropout
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt

model = Sequential()
model.add(Lambda(lambda x : x/float(255) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((75,25),(0,0))))
model.add(Conv2D(24,5,5, subsample=(2,2), border_mode='valid',  activation="elu"))
model.add(Dropout(0.5))
model.add(Conv2D(36,5,5, subsample=(2,2), border_mode='valid', activation="elu"))
model.add(Dropout(0.5))
model.add(Conv2D(48,5,5, subsample=(2,2), border_mode='valid',activation="elu"))
model.add(Dropout(0.5))
model.add(Conv2D(64,3,3, border_mode='valid',activation="elu"))
model.add(Dropout(0.5))
model.add(Conv2D(64,2,2, border_mode='valid',activation="elu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=140,  batch_size=800)
model.save('model10'
           '.h5')


### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()