import pdb
import sys
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import ModelCheckpoint
from PIL import Image
import matplotlib.pyplot as plt
from binary_nets import *


def train_net1():
	# Instante HDF5Matrix for the training set
	X_train = HDF5Matrix('data_net1_small.h5', 'data', start=0, end=250)
	y_train = HDF5Matrix('data_net1_small.h5', 'labels', start=0, end=250)

	# Instante HDF5Matrix for the test set
	X_test = HDF5Matrix('data_net1_small.h5', 'data', start=250, end=300)
	y_test = HDF5Matrix('data_net1_small.h5', 'labels', start=250, end=300)

	# Zero center
	means = np.mean(X_train, axis = 0)
	X_train -= means
	X_test -= means

	# Create model
	layer, model = build_net_1(Input(shape=(X_train.shape[1], X_train.shape[2], 3)))
	print model.summary()

	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

 	# Checkpoint (for saving the weights)
	filepath = 'Weights/weights_binary_net1.hdf5'
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, 
			save_weights_only=True, mode='max')
	callbacks_list = [checkpoint]
	
	# Train model (and save the weights)
	model.fit(X_train, y_train,
			batch_size=32,
			epochs=30,
			validation_data=(X_test, y_test),
			shuffle='batch', # Have to use shuffle='batch' or False with HDF5Matrix
			verbose=0, 
			callbacks=callbacks_list)

	# Evaluate model
	scores = model.evaluate(X_test, y_test, batch_size=32, verbose=0)
	print("%s:\t%.4f" % (model.metrics_names[0], scores[0]))
	print("%s:\t%.2f%%" % (model.metrics_names[1], scores[1]*100))


def train_net2():
	pass


def train_net3():
	pass


if __name__ == '__main__':
    try:
        arg1 = sys.argv[1]
    except IndexError:
        print("Command line argument missing. Usage: train_binary_nets.py <net number>")
        sys.exit(1)
    if arg1 == '1':
    	train_net1()
    elif arg1 == '2':
    	train_net2()
    elif arg1 == '3':
    	train_net3()
    else:
    	print("Wrong command line argument. Must be a value between 1-3.")


