from keras.utils.io_utils import HDF5Matrix
from calibration_nets import *
from global_variables import *
import numpy as np
import pdb

def predict_calib_net1(X_test):

    means = np.load(PATH + 'Datasets/means_calib1.npy')
    X_test -= means

    # Create model
    model = calibration_net_1(Input(shape=(X_test.shape[1], X_test.shape[2], 3)),N_CALIBRATION_TRANSFORMATIONS)

    # Load model
    model.load_weights(PATH + 'Weights/weights_calibration_net1.hdf5')

    # Predict values
    prediction  = model.predict(X_test)
    labels = []
    for i in range(X_test.shape[0]):
        lab = []
        for j in range(len(prediction[i,:])):
            if prediction[i,j] > CALIBRATION_THRESHOLD:
                lab.append(j)
        labels.append(lab)

    return labels

def predict_calib_net2(X_test):

    means = np.load(PATH + 'Datasets/means_calib2.npy')
    X_test -= means

    # Create model
    model = calibration_net_2(Input(shape=(X_test.shape[1], X_test.shape[2], 3)),N_CALIBRATION_TRANSFORMATIONS)

    # Load model
    model.load_weights(PATH + 'Weights/weights_calibration_net2.hdf5')

    # Predict values
    prediction  = model.predict(X_test)
    labels = []
    for i in range(X_test.shape[0]):
        lab = []
        for j in range(len(prediction[i,:])):
            if prediction[i,j] > CALIBRATION_THRESHOLD:
                lab.append(j)
        labels.append(lab)

    return labels

def predict_calib_net3(X_test):

    means = np.load(PATH + 'Datasets/means_calib3.npy')
    X_test -= means

    # Create model
    model = calibration_net_3(Input(shape=(X_test.shape[1], X_test.shape[2], 3)),N_CALIBRATION_TRANSFORMATIONS)

    # Load model
    model.load_weights(PATH + 'Weights/weights_calibration_net3.hdf5')

    # Predict values
    prediction  = model.predict(X_test)
    labels = []
    for i in range(X_test.shape[0]):
        lab = []
        for j in range(len(prediction[i,:])):
            if prediction[i,j] > CALIBRATION_THRESHOLD:
                lab.append(j)
        labels.append(lab)

    return labels

##################
#       RUN      #
##################
if __name__ == '__main__':
    samples = np.random.randn(20,25,25,3)
    pred = predict_calib_net1(samples)
