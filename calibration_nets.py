import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras.utils import np_utils

def calibration_net_1(input_img, N=45):
	conv2d = Conv2D(16, (3, 3), input_shape=input_img.shape, strides=1, padding='same', dilation_rate=1, 
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None) (input_img)
	max_pooling2d = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv2d)
	flatten = Flatten()(max_pooling2d)
	dense1 = Dense(128, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(flatten)
	dense2 = Dense(N, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(dense1)
	return Model(inputs=input_img, outputs=dense2)


def calibration_net_2(input_img, N=45):
	conv2d = Conv2D(32, (5, 5), input_shape=input_img.shape, strides=1, padding='same', dilation_rate=1, 
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None) (input_img)
	max_pooling2d = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv2d)
	flatten = Flatten()(max_pooling2d)
	dense1 = Dense(64, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(flatten)
	dense2 = Dense(N, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(dense1)
	return Model(inputs=input_img, outputs=dense2)


"""Testing"""
if __name__ == "__main__":
	input_size = 25
	input_img = Input(shape=(input_size, input_size, 3))

	print("\nnet_1\n")
	m_1 = calibration_net_1(input_img)
	print (m_1.summary())

	input_size = 50
	input_img = Input(shape=(input_size, input_size, 3))
	print ("\nnet_2\n")
	m_2 = calibration_net_2(input_img)
	print (m_2.summary())