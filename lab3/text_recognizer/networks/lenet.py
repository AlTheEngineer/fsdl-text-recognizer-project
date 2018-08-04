from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D
from tensorflow.keras.models import Sequential, Model


def lenet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:
    
    image_height, image_width = input_shape
    
    num_classes = output_shape[0]
    
    image_input = Input(shape=input_shape)
    
    image_reshaped = Reshape((image_height, image_width, 1))(image_input)

    
    # Conv1
    X = Conv2D(64, (5, 5), padding='same', name = 'conv1')(image_reshaped)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)    
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)

    # Conv2
    X = Conv2D(128, (2, 2), padding='same', name = 'conv2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv2')(X)


    # Output FC layers
    X = Flatten()(X)
    X = Dense(1024, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(512, activation='relu')(X)
    X = BatchNormalization()(X)    
    X = Dense(128, activation='relu')(X)
    
    preds = Dense(num_classes, activation='softmax')(X)
    
    
    # Create model
    model = Model(inputs=image_input, outputs=preds)

    return model

