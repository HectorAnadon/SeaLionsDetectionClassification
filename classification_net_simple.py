from keras.utils.io_utils import HDF5Matrix
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import regularizers
from keras.utils import np_utils
from global_variables import * 
from make_datasets import *

def build_classif_net(input_img):
    """Build a simple classification net
    """
    print input_img.shape

    conv2d_1 = Conv2D(16, (5, 5), input_shape=input_img.shape, strides=1, padding='same', dilation_rate=1, 
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=regularizers.l2(REGULARIZATION_MULTICLASS), bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None) (input_img)
    conv2d_1 = BatchNormalization()(conv2d_1)
    max_pooling2d_1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv2d_1)

    conv2d_2 = Conv2D(32, (5, 5), strides=1, padding='same', dilation_rate=1, 
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=regularizers.l2(REGULARIZATION_MULTICLASS), bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None) (max_pooling2d_1)
    conv2d_2 = BatchNormalization()(conv2d_2)
    max_pooling2d_2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv2d_2)

    conv2d_3 = Conv2D(64, (5, 5), strides=1, padding='same', dilation_rate=1, 
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=regularizers.l2(REGULARIZATION_MULTICLASS), bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None) (max_pooling2d_2)
    conv2d_3 = BatchNormalization()(conv2d_3)
    max_pooling2d_3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv2d_3)

    flatten = Flatten()(max_pooling2d_3)
    dense_1 = Dense(100, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=regularizers.l2(REGULARIZATION_MULTICLASS), bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(flatten)
    dense_1 = BatchNormalization()(dense_1)
    dense_2 = Dense(50, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=regularizers.l2(REGULARIZATION_MULTICLASS), bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(dense_1)
    dense_2 = BatchNormalization()(dense_2)
    output = Dense(5, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=regularizers.l2(REGULARIZATION_MULTICLASS), bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(dense_2)
    model = Model(inputs=input_img, outputs=output)
    return model  


def train_classif_net():
    # Load data
    X_data = HDF5Matrix(PATH + 'Datasets/data_positive_net4_small.h5', 'data')
    y_data = HDF5Matrix(PATH + 'Datasets/data_positive_net4_small.h5', 'labels')
    # Split into training and validation sets
    X_train, y_train, X_test, y_test = split_data(X_data, y_data, TRAIN_SPLIT)
    # Zero center
    means = np.mean(X_train, axis = 0)
    X_train -= means
    X_test -= means
    # Create model

    model = build_classif_net(Input(shape=(ORIGINAL_WINDOW_DIM, ORIGINAL_WINDOW_DIM, 3)))
    print(model.summary())
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    # # Checkpoint (for saving the weights)
    # filepath = 'Weights/weights_binary_net4.hdf5'
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, 
    #         save_weights_only=True, mode='max')
    # callbacks_list = [checkpoint]
    # Train model (and save the weights)
    model.fit(X_train, y_train,
            batch_size=32,
            epochs=30,
            validation_data=(X_test, y_test),
            shuffle='batch', # Have to use shuffle='batch' or False with HDF5Matrix
            verbose=1)
            #callbacks=callbacks_list)

def predict_classif_net():
    pass



