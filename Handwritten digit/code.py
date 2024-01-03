# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 23:38:12 2024

@author: theli
"""
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist
#training for training model, teeting is used to assess model
#load data is already split up
#x is image, y is classification
(x_train,y_train), (x_test,y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)

x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# #flatten the 2 dimension into 1 dimension layer
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# #output layer
# model.add(tf.keras.layers.Dense(10, activation='softmax'))
# #softmax makes sure all the outputs add up to 1, each of the 10 digit neurons has a value from 0 to 1, gives us probability of each digit to be right answer

# model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs = 3)
# model.save('handwritten.model')

model = tf.keras.models.load_model('handwritten.model')

# loss, accuracy = model.evaluate(x_test,y_test)
# print(loss)
# print(accuracy)


image_number = int(input("give a number"))
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"this digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number +=1