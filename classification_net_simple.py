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

def build_classif_net():
    """Build a simple classification net
    """
    model = Sequential()

    model.add(Conv2D(32, (5, 5), input_shape=(ORIGINAL_WINDOW_DIM,ORIGINAL_WINDOW_DIM,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
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



