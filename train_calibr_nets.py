import pdb
from keras.utils.io_utils import HDF5Matrix
from PIL import Image
import matplotlib.pyplot as plt
from binary_nets import *
from sklearn import model_selection as ms

def train_calibr_net1():

    X_data = HDF5Matrix('data_callib1_small.h5', 'data')
    y_data = HDF5Matrix('data_callib1_small.h5', 'labels')
    print X_data.shape
    print y_data.shape

    pdb.set_trace()
    X_train, X_test, y_train, y_test = ms.train_test_split(X_data, y_data, train_size=0.7, random_state=0)
    pdb.set_trace()

    # Zero center
    means = np.mean(X_train, axis = 0)
    X_train -= means
    X_test -= means

    layer, model = build_net_1(Input(shape=(25, 25, 3)))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    print model.summary()
    # Note: you have to use shuffle='batch' or False with HDF5Matrix
    #model.fit(X_train, y_train, batch_size=32, shuffle='batch')

    model.fit(X_train, y_train,
	          batch_size=32,
	          epochs=30,
	          verbose=1,
	          validation_data=(X_test, y_test),
	          shuffle='batch')

    model.evaluate(X_test, y_test, batch_size=32)

train_calibr_net1()