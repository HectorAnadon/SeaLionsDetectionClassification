import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras.utils import np_utils

def build_12_net():
    # input image dimensions
    img_rows, img_cols = 25, 25
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(img_rows, img_cols, 3), strides=1, padding='same', dilation_rate=1, 
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))
    model.add(Flatten())
    model.add(Dense(2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    return model


# def build_24_net():
#     # input image dimensions
#     img_rows, img_cols = 50, 50

#     model = Sequential()

#     model.add(Conv2D(64, (5, 5), input_shape=(img_rows, img_cols, 3), strides=1, padding='same', dilation_rate=1, 
#         activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
#         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
#         activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

#     model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))

#     flatten = Flatten()  # check that this is 128 * 6 * 6

#     model_12_net = build_12_net()
#     flatten_12 = model_12_net.layers[-2]
#     print flatten_12

#     model.add(concatenate([flatten, flatten_12]))

#     model.add(Dense(2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
#         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
#         activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    
#     return model

def build_50_net():
    input_img_50 = Input(shape=(50, 50, 3))

    conv2d_50 = Conv2D(64, (5, 5), input_shape=input_img_50.shape, strides=1, padding='same', dilation_rate=1, 
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None) (input_img_50)

    max_pooling2d_50 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv2d_50)

    flatten_50 = Flatten()(max_pooling2d_50)


    input_img_25 = Input(shape=(25, 25, 3))
    conv2d_25 = Conv2D(16, (5, 5), input_shape=input_img_25.shape, strides=1, padding='same', dilation_rate=1, 
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None) (input_img_25)
    max_pooling2d_25 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv2d_25)
    flatten_25 = Flatten()(max_pooling2d_25)


    flatten_merged = concatenate([flatten_50, flatten_25])

    dense = Dense(2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(flatten_merged)

    model = Model(inputs = [input_img_50 , input_img_25] , outputs = dense)

    return model



"""Testing"""

# print "\n\t12_net\n"
# m12 = build_12_net()
# print m12.summary()
# print "FC layer:\n", m12.layers[-1], m12.layers[-1].get_config()

print "\n\t50_net\n"
m50 = build_50_net()
print m50.summary()
print "FC layer:\n", m50.layers[-1], m50.layers[-1].get_config()



