import keras
from keras.layers import Conv2D, MaxPooling2D, Lambda, Input, Dense, Flatten, BatchNormalization
from keras.models import Model
from keras.layers.core import Dropout
from keras import optimizers
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau,TensorBoard

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.preprocessing import OneHotEncoder
from keras.datasets import cifar10
# import cv2
# import gc
import numpy as np

# URL http://blog.neko-ni-naritai.com/entry/2018/04/07/115504

print(keras.__version__)

# Prepare data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
enc = OneHotEncoder()
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.fit_transform(y_test).toarray()

# inputs = Input(shape=(224, 224, 3))
inputs = Input(shape=(32, 32, 3))
# Due to memory limitation, images will resized on-the-fly.
x = Lambda(lambda image: tf.image.resize_images(image, (224, 224)))(inputs)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = BatchNormalization()(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = BatchNormalization()(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = BatchNormalization()(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = BatchNormalization()(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block5_pool')(x)
flattened = Flatten(name='flatten')(x)
x = Dense(4096, activation='relu', name='fc1')(flattened)
x = Dropout(0.5, name='dropout1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dropout(0.5, name='dropout2')(x)

# CIFAR10は10クラスなので出力の数が違う
predictions = Dense(10, activation='softmax', name='predictions')(x)

BATCH_SIZE = 256
sgd = optimizers.SGD(lr=0.01,
                     momentum=0.9,
                     decay=5e-4)#, nesterov=False)

model = Model(inputs=inputs, outputs=predictions)


model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
rlop = ReduceLROnPlateau(monitor='val_acc',
                         factor=0.1,
                         patience=5,
                         verbose=1,
                         mode='auto',
                         epsilon=0.0001,
                         cooldown=0,
                         min_lr=0.00001)

 
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=30, verbose=1,
              callbacks=[rlop], validation_data=(x_test, y_test))

y_pred = model.predict(x_test, verbose=1)

print(confusion_matrix(np.argmax(y_test, 1), np.argmax(y_pred, 1)))
print(classification_report(np.argmax(y_test, 1), np.argmax(y_pred, 1)))

