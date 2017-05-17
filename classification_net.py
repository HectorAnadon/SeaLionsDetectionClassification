from keras.applications.vgg19 import VGG19
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Model
from global_variables import *

def classification_net():
    base_model = VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=(ORIGINAL_WINDOW_DIM, ORIGINAL_WINDOW_DIM, 3))
    print(base_model.summary())
    keras_input = Input(shape=(ORIGINAL_WINDOW_DIM, ORIGINAL_WINDOW_DIM, 3),name = 'image_input')
    base_model = base_model(keras_input)

    flatten = Flatten(name='flatten')(base_model)
    dense1 = Dense(4096, activation='relu')(flatten)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(4096, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    output = Dense(5, activation='softmax')(dropout2)
    model = Model(input=keras_input, output=output)

    return model


