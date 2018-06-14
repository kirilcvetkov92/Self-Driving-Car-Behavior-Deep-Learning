import csv
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.convolutional import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers import Dropout
from keras.layers import BatchNormalization


from utils import generator, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

BATCH_SIZE = 32
EPOCHS = 20

samples = []
distribution=[]

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        distribution.append(float(line[3]))

#plot data
n, bins, patches = plt.hist(np.array(distribution), bins=20)
plt.show()
del distribution[:]

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE, is_training=False)


model = Sequential()
# Preprocess  (it will be performed parallel on gpu)
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS)))

# Convolution layers with batch normalization in between
model.add(Conv2D(24,5,5, subsample=(2,2), border_mode='valid',  activation="elu"))
model.add(BatchNormalization())

model.add(Conv2D(36,5,5, subsample=(2,2), activation="elu"))
model.add(BatchNormalization())

model.add(Conv2D(48,5,5, subsample=(2,2),activation="elu"))
model.add(BatchNormalization())

model.add(Conv2D(64,3,3, activation="elu"))
model.add(BatchNormalization())

model.add(Conv2D(64,3,3,activation="elu"))
model.add(BatchNormalization())

# Dense layers with batch normalization and dropout in between
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(50, activation='elu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(10, activation='elu'))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')

#Get model training checkpoints
checkpoint = ModelCheckpoint('checkpoints/model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=False,
                                 mode='auto')


#model = load_model('checkpoints/model.h5')

history_object = model.fit_generator(train_generator,
                                     len(train_samples)//BATCH_SIZE,
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples)//BATCH_SIZE,
                                     nb_epoch=EPOCHS,
                                     callbacks=[checkpoint])
model.save('model'
           '.h5')


# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()