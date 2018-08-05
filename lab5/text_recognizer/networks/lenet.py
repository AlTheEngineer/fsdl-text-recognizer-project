from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D, Reshape, BatchNormalization
from tensorflow.keras.models import Sequential, Model


def lenet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:
    num_classes = output_shape[0]

    image_height, image_width, nc = input_shape
    
    num_classes = output_shape[0]
    
    image_input = Input(shape=input_shape)
    
    image_reshaped = Reshape((image_height, image_width, 1))(image_input)

    model = Sequential()
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape=input_shape))
        input_shape = (input_shape[0], input_shape[1], 1)
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

"""LeNet
    ##### Your code below (Lab 2)
    model = Sequential()
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape=input_shape))
        input_shape = (input_shape[0], input_shape[1], 1)
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    ##### Your code above (Lab 2)
"""

"""Puigcerver's Paper
# Stage 1
    X = Conv2D(16, (3, 3), padding='valid', name = 'conv1')(image_reshaped)
    X = BatchNormalization(name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)

    
    X = Conv2D(32, (3, 3), padding='valid', name = 'conv2')(image_reshaped)
    X = BatchNormalization(name = 'bn_conv2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    
    X = Conv2D(48, (3, 3), padding='valid', name = 'conv3')(image_reshaped)
    X = BatchNormalization(name = 'bn_conv3')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = Dropout(0.2)(X)
    
    
    X = Conv2D(64, (3, 3), padding='valid', name = 'conv4')(image_reshaped)
    X = BatchNormalization(name = 'bn_conv4')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = Dropout(0.2)(X)

    X = Conv2D(80, (3, 3), padding='valid', name = 'conv5')(X)    
    X = BatchNormalization(axis = 3, name = 'bn_conv5')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = Dropout(0.2)(X)
    
    X = Flatten()(X)
    # output layer
    preds = Dense(num_classes, activation='softmax')(X)
    
    
    # Create model
    model = Model(inputs=image_input, outputs=preds)
"""