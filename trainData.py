import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

IMAGE_SIZE = 50

featureSet = pickle.load(open('featureSet.pickle', 'rb'))
featureSet = np.array(featureSet).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1) / 255

labelSet = pickle.load(open('labelSet.pickle', 'rb'))
labelSet = np.array(labelSet)

model = Sequential()

model.add(Conv2D(128, (3,3), input_shape = featureSet.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(128, (3,3), input_shape = featureSet.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
model.fit(featureSet, labelSet, batch_size = 4, epochs = 10, validation_split = 0.1)
model.save('maskDetector.model')
