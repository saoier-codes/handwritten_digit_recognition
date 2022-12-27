#import libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#loading datasets
mnist = tf.keras.datasets.mnist

#to training data and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normailizing
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#create a model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu')
model.add(tf.keras.layers.Dense(10, activation='softmax'))

#compile model
model.compile(optimizer='adam', loss='sparse_categorial_crossentropy', metrics=['accuracy'] )

model.fit(x_train,y_train, epochs=5)

model.save('handwritten.model')
