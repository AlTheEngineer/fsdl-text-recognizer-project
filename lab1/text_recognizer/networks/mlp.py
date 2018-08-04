from typing import Tuple
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Input, Add, Activation, ZeroPadding2D, BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Reshape
from tensorflow.keras.applications import resnet50

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
        
    return X



def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (s, s), padding = 'valid', name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '2c')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s, s), padding = 'valid', name = conv_name_base + '1')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
        
    return X

def mlp(input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        layer_size: int=128,
        dropout_amount: float=0.2,
        num_layers: int=3) -> Model:
    """
    Simple multi-layer perceptron: just fully-connected layers with dropout between them, with softmax predictions.
    Creates num_layers layers.
    """
    image_height, image_width = input_shape
    num_classes = output_shape[0]
    
    image_input = Input(shape=input_shape)
    
    image_reshaped = Reshape((image_height, image_width, 1))(image_input)

    
    # Stage 1
    X = Conv2D(64, (5, 5), padding='same', name = 'conv1')(image_reshaped)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)    
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)


    X = Conv2D(128, (2, 2), padding='same', name = 'conv2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv2')(X)


    # output layer
    X = Flatten()(X)
    X = Dense(1024, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(512, activation='relu')(X)
    X = BatchNormalization()(X)    
    X = Dense(128, activation='relu')(X)
    
    preds = Dense(num_classes, activation='softmax')(X)
    
    
    # Create model
    model = Model(inputs=image_input, outputs=preds)
    
    ##### Your code above (Lab 1)

    return model

"""MLP
    model = Sequential()
    # Don't forget to pass input_shape to the first layer of the model
    ##### Your code below (Lab 1)
    model.add(Flatten(input_shape=input_shape))
    # use for loop for layers and list of layer sizes
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_shape[0], activation='softmax'))
"""


"""LeNet
    X = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(image_height, image_width, 1))(image_reshaped)
    X = Conv2D(64, (3, 3), activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(0.2)(X)
    X = Flatten()(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.2)(X)
    preds = Dense(num_classes, activation='softmax')(X)
    # Create model
    model = Model(inputs=image_input, outputs=preds)
"""

"""ResNet50
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(image_reshaped)
    
    # Stage 1
    X = Conv2D(6, (3, 3), strides = (2, 2), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [6, 6, 18], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [6, 6, 18], stage=2, block='b')
    X = identity_block(X, 3, [6, 6, 18], stage=2, block='c')


    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [12, 12, 36], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [12, 12, 36], stage=3, block='b')
    X = identity_block(X, 3, [12, 12, 36], stage=3, block='c')
    X = identity_block(X, 3, [12, 12, 36], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [24, 24, 72], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [24, 24, 72], stage=4, block='b')
    X = identity_block(X, 3, [24, 24, 72], stage=4, block='c')
    X = identity_block(X, 3, [24, 24, 72], stage=4, block='d')
    X = identity_block(X, 3, [24, 24, 72], stage=4, block='e')
    X = identity_block(X, 3, [24, 24, 72], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [48, 48, 144], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [48, 48, 144], stage=5, block='b')
    X = identity_block(X, 3, [48, 48, 144], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    # X = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)
    
    # output layer
    X = Flatten()(X)
    preds = Dense(num_classes, activation='softmax')(X)
"""