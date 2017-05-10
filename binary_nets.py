import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras.utils import np_utils


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
    dense = Dense(2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(flatten)
    model = Model(inputs=input_img, outputs=dense)

    return flatten, model    

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
    # First net
    input_img_1 = Input(shape=(25, 25, 3)) # TODO change this to proper image
    flatten_1, _ = build_net_1(input_img_1)
    # Concatenate
    flatten_merged = concatenate([flatten_2, flatten_1])
    dense = Dense(2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(flatten_merged)
    model = Model(inputs = [input_img, input_img_1] , outputs = dense)

    return flatten_merged, model




"""Testing"""

input_size = 25
input_img = Input(shape=(input_size, input_size, 3))

print "\nnet_1\n"
_, m_1 = build_net_1(input_img)
print m_1.summary()

input_size = 50
input_img = Input(shape=(input_size, input_size, 3))
print "\nnet_2\n"
_, m_2 = build_net_2(input_img)
print m_2.summary()


