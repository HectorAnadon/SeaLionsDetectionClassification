import numpy as np
# np.random.seed(1337)  # for reproducibility

from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization


def build_net_1(input_img):
    """Build the first convolutional network (e.g. 25-net)

    input_img - Input() tensor
    """
    conv2d = Conv2D(16, (5, 5), input_shape=input_img.shape, strides=1, padding='same', dilation_rate=1, 
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None) (input_img)
    max_pooling2d = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv2d)
    flatten = Flatten()(max_pooling2d)
    dense = Dense(16, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(flatten)
    output = Dense(2, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(dense)
    model = Model(inputs=input_img, outputs=output)

    return dense, model    

def build_net_2(input_img):
    """Build the second convolutional network (e.g. 50-net)

    input_img - Input() tensor
    """
    # Second net
    conv2d = Conv2D(64, (5, 5), input_shape=input_img.shape, strides=1, padding='same', dilation_rate=1, 
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None) (input_img)
    max_pooling2d = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv2d)
    flatten_2 = Flatten()(max_pooling2d)
    dense = Dense(128, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(flatten_2)
    # First net
    size = input_img.get_shape().as_list()[1]
    size = int(size / 2)
    input_img_1 = Input(shape=(size, size, 3))
    network_1, _ = build_net_1(input_img_1)
    # Concatenate
    merged = concatenate([dense, network_1])
    output = Dense(2, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(merged)
    model = Model(inputs = [input_img, input_img_1] , outputs = output)

    return merged, model

def build_net_3(input_img):
    """Build the second convolutional network (e.g. 100-net)

    input_img - Input() tensor
    """
    # Third net
    conv2d_1_3 = Conv2D(64, (5, 5), input_shape=input_img.shape, strides=1, padding='same', dilation_rate=1, 
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None) (input_img)
    batch_norm_1 = BatchNormalization()(conv2d_1_3)
    max_pooling2d_1_3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(batch_norm_1)
    batch_norm_2 = BatchNormalization()(max_pooling2d_1_3)
    conv2d_2_3 = Conv2D(64, (5, 5), strides=1, padding='same', dilation_rate=1, 
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None) (batch_norm_2)
    max_pooling2d_2_3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv2d_2_3)
    flatten_3 = Flatten()(max_pooling2d_2_3)
    dense_3 = Dense(256, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(flatten_3)
    
    # Second net
    size = input_img.get_shape().as_list()[1]
    size = int(size / 2)
    input_img_2 = Input(shape=(size, size, 3)) 
    conv2d = Conv2D(64, (5, 5), input_shape=input_img_2.shape, strides=1, padding='same', dilation_rate=1, 
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None) (input_img_2)
    max_pooling2d = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv2d)
    flatten_2 = Flatten()(max_pooling2d)
    dense = Dense(128, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(flatten_2)
    # First net
    size = int(size/ 2)
    input_img_1 = Input(shape=(size, size, 3)) 
    network_1, _ = build_net_1(input_img_1)
    # Concatenate 2 and 1
    network_2 = concatenate([dense, network_1])

    # Concatenate 3 and 2
    flatten_merged = concatenate([dense_3, network_2])
    output = Dense(2, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(flatten_merged)
    return Model(inputs = [input_img, input_img_2, input_img_1] , outputs = output)



"""Testing"""
if __name__ == "__main__":
    input_size = 25
    input_img = Input(shape=(input_size, input_size, 3))
    print("\nnet_1\n")
    _, m_1 = build_net_1(input_img)
    print (m_1.summary())

    input_size = 50
    input_img = Input(shape=(input_size, input_size, 3))
    print ("\nnet_2\n")
    _, m_2 = build_net_2(input_img)
    print (m_2.summary())

    input_size = 100
    input_img = Input(shape=(input_size, input_size, 3))
    print ("\nnet_3\n")
    m_3 = build_net_3(input_img)
    print (m_3.summary())