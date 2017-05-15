import pdb
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import ModelCheckpoint
from PIL import Image
import matplotlib.pyplot as plt
from calibration_nets import *
from sklearn import model_selection as ms
from global_variables import *

def train_calibr_net1():
    X_data = HDF5Matrix('Datasets/data_callib1_small.h5', 'data')
    y_data = HDF5Matrix('Datasets/data_callib1_small.h5', 'labels')
    print (X_data.shape)
    print (y_data.shape)
    

    num_images = y_data.shape[0]
    train_split = TRAIN_SPLIT
    X_train = X_data[0:int(round(train_split*num_images))]
    y_train = y_data[0:int(round(train_split*num_images))]
    X_test = X_data[int(round(train_split*num_images))+1:-1]
    y_test = y_data[int(round(train_split*num_images))+1:-1]
    # Zero center
    means = np.mean(X_train, axis = 0)
    X_train -= means
    X_test -= means

    model = calibration_net_1(Input(shape=(INPUT_SIZE_NET_1, INPUT_SIZE_NET_1, 3)), N_CALIBRATION_TRANSFORMATIONS)

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # print (model.summary())
    # Note: you have to use shuffle='batch' or False with HDF5Matrix
    #model.fit(X_train, y_train, batch_size=32, shuffle='batch')

    filepath = 'Weights/weights_calibration_net1.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, 
            save_weights_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(X_train, y_train,
	          batch_size=32,
	          epochs=30,
	          verbose=1,
	          validation_data=(X_test, y_test),
	          shuffle='batch', 
              callbacks=callbacks_list)

    model.evaluate(X_test, y_test, batch_size=32)

if __name__ == '__main__':
    train_calibr_net1()