import sys, pdb
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import ModelCheckpoint, Callback
from PIL import Image
import matplotlib.pyplot as plt
from calibration_nets import *
from global_variables import *
from make_datasets import *

def train_calibr_net1():
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.metrics = []

        def on_epoch_end(self, epoch, logs={}):
            self.metrics.append(logs)
            np.save(PATH + 'Results/loss_calibration_net1.npy', np.array(self.metrics))

    """Train the first binary net and save training data means and best model weights.
    """
    # Load data
    X_data = HDF5Matrix(PATH + 'Datasets/data_calib1_pups.h5', 'data')
    y_data = HDF5Matrix(PATH + 'Datasets/data_calib1_pups.h5', 'labels')
    # Split into training and validation sets
    X_train, y_train, X_test, y_test = split_data(X_data, y_data, TRAIN_SPLIT)
    # Zero center
    means = np.mean(X_train, axis = 0)
    X_train -= means
    X_test -= means
    # Save means (for testing)
    np.save(PATH + 'Datasets/means_calib1.npy',means)
    # Create model
    model = calibration_net_1(Input(shape=(X_train.shape[1], X_train.shape[2], 3)), N_CALIBRATION_TRANSFORMATIONS)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    print(model.summary())
    history = LossHistory()
    # Checkpoint (for saving the weights)
    filepath = PATH + 'Weights/weights_calibration_net1.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, 
            save_weights_only=True, mode='max')
    callbacks_list = [checkpoint, history]
    # Train model (and save the weights)
    model.fit(X_train, y_train,
	          batch_size=32,
	          epochs=30,
	          verbose=1,
	          validation_data=(X_test, y_test),
	          shuffle='batch', # have to use shuffle='batch' or False with HDF5Matrix
              callbacks=callbacks_list)


def train_calibr_net2():
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.metrics = []

        def on_epoch_end(self, epoch, logs={}):
            self.metrics.append(logs)
            np.save(PATH + 'Results/loss_calibration_net2.npy', np.array(self.metrics))

    """Train the second binary net and save training data means and best model weights.
    """
    # Load data
    X_data = HDF5Matrix(PATH + 'Datasets/data_calib2_pups.h5', 'data')
    y_data = HDF5Matrix(PATH + 'Datasets/data_calib2_pups.h5', 'labels')
    # Split into training and validation sets
    X_train, y_train, X_test, y_test = split_data(X_data, y_data, TRAIN_SPLIT)
    # Zero center
    means = np.mean(X_train, axis = 0)
    X_train -= means
    X_test -= means
    # Save means (for testing)
    np.save(PATH + 'Datasets/means_calib2.npy',means)
    # Create model
    model = calibration_net_2(Input(shape=(X_train.shape[1], X_train.shape[2], 3)), N_CALIBRATION_TRANSFORMATIONS)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    print(model.summary())
    history = LossHistory()
    # Checkpoint (for saving the weights)
    filepath = PATH + 'Weights/weights_calibration_net2.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, 
            save_weights_only=True, mode='max')
    callbacks_list = [checkpoint, history]
    # Train model (and save the weights)
    model.fit(X_train, y_train,
              batch_size=32,
              epochs=30,
              verbose=1,
              validation_data=(X_test, y_test),
              shuffle='batch', # have to use shuffle='batch' or False with HDF5Matrix
              callbacks=callbacks_list)


def train_calibr_net3():
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.metrics = []

        def on_epoch_end(self, epoch, logs={}):
            self.metrics.append(logs)
            np.save(PATH + 'Results/loss_calibration_net3.npy', np.array(self.metrics))

    """Train the third binary net and save training data means and best model weights.
    """
    # Load data
    X_data = HDF5Matrix(PATH + 'Datasets/data_calib3_pups.h5', 'data')
    y_data = HDF5Matrix(PATH + 'Datasets/data_calib3_pups.h5', 'labels')
    # Split into training and validation sets
    X_train, y_train, X_test, y_test = split_data(X_data, y_data, TRAIN_SPLIT)
    # Zero center
    means = np.mean(X_train, axis = 0)
    X_train -= means
    X_test -= means
    # Save means (for testing)
    np.save(PATH + 'Datasets/means_calib3.npy',means)
    # Create model
    model = calibration_net_3(Input(shape=(X_train.shape[1], X_train.shape[2], 3)), N_CALIBRATION_TRANSFORMATIONS)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    print(model.summary())
    history = LossHistory()
    # Checkpoint (for saving the weights)
    filepath = PATH + 'Weights/weights_calibration_net3.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, 
            save_weights_only=True, mode='max')
    callbacks_list = [checkpoint, history]
    # Train model (and save the weights)
    model.fit(X_train, y_train,
              batch_size=32,
              epochs=30,
              verbose=1,
              validation_data=(X_test, y_test),
              shuffle='batch', # have to use shuffle='batch' or False with HDF5Matrix
              callbacks=callbacks_list)


"""Testing"""
if __name__ == '__main__':
    try:
        arg1 = sys.argv[1]
    except IndexError:
        print("Command line argument missing. Usage: train_calibr_nets.py <net number>")
        sys.exit(1)
    if arg1 == '1':
        train_calibr_net1()
    elif arg1 == '2':
        train_calibr_net2()
    elif arg1 == '3':
        train_calibr_net3()
    else:
        print("Wrong command line argument. Must be a value between 1-3.")



