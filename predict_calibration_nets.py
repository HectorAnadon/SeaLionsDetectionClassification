from keras.utils.io_utils import HDF5Matrix
from calibration_nets import *
import numpy as np

def predict_calib_net1(X_test, N = 9):

    #means = HDF5Matrix('means_calib1.h5', 'data')
    means = 100
    X_test -= means

    # Create model
    model = calibration_net_1(Input(shape=(X_test.shape[1], X_test.shape[2], 3)),N)

    # Load model
    model.load_weights('./Weights/weights_calibration_net1.hdf5')

    # Predict values
    prediction  = model.predict(X_test)
    label = np.argmax(prediction)

    return label

##################
#       RUN      #
##################

samples = np.random.randn(20,25,25,3)
pred = predict_calib_net1(samples, N = 9)