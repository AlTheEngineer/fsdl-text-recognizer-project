from typing import Tuple
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Input, Add, Activation, ZeroPadding2D, BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Reshape

def resnet(input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...]) -> Model:
    """
    Simple multi-layer perceptron: just fully-connected layers with dropout between them, with softmax predictions.
    Creates num_layers layers.
    """
    image_height, image_width, nc = input_shape
    
    num_classes = output_shape[0]
    
    image_input = Input(shape=input_shape)
    
    image_reshaped = Reshape((image_height, image_width, 1))(image_input)

    
    # Stage 1
    X = Conv2D(28, (5, 5), padding='same', name = 'conv1')(image_reshaped)
    X = Activation('relu')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)

    # feature map shape = (28, 28, 64)
    
    X = convolutional_block(X, 5, [64, 64, 64], 1, "a")
    X = Activation('relu')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv2')(X)
    # feature map shape = (14, 14, 64)
    X = convolutional_block(X, 3, [128, 128, 128], 2, "a")
    X = Activation('relu')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv3')(X)
    # feature map shape = (8, 8, 128)
    X = AveragePooling2D((2, 2), strides=(2, 2))(X)
    # feature map shape = (4, 4, 128)
    #X = Dropout(0.25)(X)
    
    X = Flatten()(X)
    X = Dense(256, activation='relu')(X)
    #X = Dropout(0.25)(X)
    X = Dense(128, activation='relu')(X)
    #X = Dropout(0.5)(X)
    
    preds = Dense(num_classes, activation='softmax')(X)
    
    # Create model
    model = Model(inputs=image_input, outputs=preds)

    return model


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block
    
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
    
    # Retrieve Filters (F3 should be equal to number of channels of the input map)
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
    Implementation of the convolutional block
    
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
    X = Activation('relu')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '2c')(X)
    
    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s, s), padding = 'valid', name = conv_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    
    return X

"""WorkingTemplate
    # Stage 1
    X = Conv2D(28, (3, 3), padding='same', name = 'conv1')(image_reshaped)
    X = Activation('relu')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    # feature map shape = (28, 28, 64)
    X = Conv2D(64, (3,3), activation='relu')(X)
    # feature map shape = (8, 8, 128)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    # feature map shape = (4, 4, 128)
    X = Dropout(0.25)(X)

    
    X = Flatten()(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.2)(X)
"""