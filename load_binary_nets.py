import pdb
import sys
from keras.utils.io_utils import HDF5Matrix
from binary_nets import *
from make_datasets import *


def load_binary_net1():
	""" Load binary_net1 model and check validation accuracy (should be same as the best
		validation accuracy when the model was trained).
	"""
	# Load data
	X_data = HDF5Matrix(PATH + 'Datasets/data_net1_small.h5', 'data')
	y_data = HDF5Matrix(PATH + 'Datasets/data_net1_small.h5', 'labels')
	# Split into training and validation sets (same used during training)
	X_train, y_train, X_test, y_test = split_data(X_data, y_data, TRAIN_SPLIT)
	# Zero center
	means = np.load(PATH + 'Datasets/means_net1.npy')
	X_test -= means
	# Create model
	layer, model = build_net_1(Input(shape=(X_train.shape[1], X_train.shape[2], 3)))
	print(model.summary())
	# Load weights
	model.load_weights(PATH + 'Weights/weights_binary_net1.hdf5')
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
	# Evaluate model (Check validation accuracy)
	scores = model.evaluate(X_test, y_test, batch_size=32, verbose=0)
	print("%s:\t%.4f" % (model.metrics_names[0], scores[0]))
	print("%s:\t%.2f%%" % (model.metrics_names[1], scores[1]*100))


def load_binary_net2():
	""" Load binary_net2 model and check validation accuracy (should be same as the best
		validation accuracy when the model was trained).
	"""
	# Load data (current net)
	X_data = HDF5Matrix(PATH + 'Datasets/data_net2_small.h5', 'data')
	y_data = HDF5Matrix(PATH + 'Datasets/data_net2_small.h5', 'labels')
	# Split into training and validation sets (same used during training)
	X_train, y_train, X_test, y_test = split_data(X_data, y_data, TRAIN_SPLIT)
	# Zero center
	means = np.load(PATH + 'Datasets/means_net2.npy')
	X_test -= means
	# Load data (previous net)
	X_data = HDF5Matrix(PATH + 'Datasets/data_net1_small.h5', 'data')
	y_data = HDF5Matrix(PATH + 'Datasets/data_net1_small.h5', 'labels')
	# Split into training and validation sets
	X_train_prev, y_train_prev, X_test_prev, y_test_prev = split_data(X_data, y_data, TRAIN_SPLIT)
	# Zero center
	means = np.load(PATH + 'Datasets/means_net1.npy')
	X_test_prev -= means
	# Create model
	layer, model = build_net_2(Input(shape=(X_train.shape[1], X_train.shape[2], 3)))
	print(model.summary())
	# Load weights
	model.load_weights(PATH + 'Weights/weights_binary_net2.hdf5')
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
	# Evaluate model (Check validation accuracy)
	scores = model.evaluate([X_test, X_test_prev], y_test, batch_size=32, verbose=0)
	print("%s:\t%.4f" % (model.metrics_names[0], scores[0]))
	print("%s:\t%.2f%%" % (model.metrics_names[1], scores[1]*100))


def load_binary_net3():
	""" Load binary_net3 model and check validation accuracy (should be same as the best
		validation accuracy when the model was trained).
	"""
	# Load data (current net)
	X_data = HDF5Matrix(PATH + 'Datasets/data_net3_small.h5', 'data')
	y_data = HDF5Matrix(PATH + 'Datasets/data_net3_small.h5', 'labels')
	# Split into training and validation sets (same used during training)
	X_train, y_train, X_test, y_test = split_data(X_data, y_data, TRAIN_SPLIT)
	# Zero center
	means = np.load(PATH + 'Datasets/means_net3.npy')
	X_test -= means
	# Load data (2nd net)
	X_data = HDF5Matrix(PATH + 'Datasets/data_net2_small.h5', 'data')
	y_data = HDF5Matrix(PATH + 'Datasets/data_net2_small.h5', 'labels')
	# Split into training and validation sets
	X_train2, y_train2, X_test2, y_test2 = split_data(X_data, y_data, TRAIN_SPLIT)
	# Zero center
	means = np.load(PATH + 'Datasets/means_net2.npy')
	X_train2 -= means
	X_test2 -= means
	# Load data (1st net)
	X_data = HDF5Matrix(PATH + 'Datasets/data_net1_small.h5', 'data')
	y_data = HDF5Matrix(PATH + 'Datasets/data_net1_small.h5', 'labels')
	# Split into training and validation sets
	X_train1, y_train1, X_test1, y_test1 = split_data(X_data, y_data, TRAIN_SPLIT)
	# Zero center
	means = np.load(PATH + 'Datasets/means_net1.npy')
	X_train1 -= means
	X_test1 -= means
	# Check the labels are the same
	assert np.array_equal(y_train, y_train2) and np.array_equal(y_train, y_train1)

	# Create model
	model = build_net_3(Input(shape=(X_train.shape[1], X_train.shape[2], 3)))
	print(model.summary())
	# Load weights
	model.load_weights(PATH + 'Weights/weights_binary_net3.hdf5')
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
	# Evaluate model (Check validation accuracy)
	scores = model.evaluate([X_test, X_test2, X_test1], y_test, batch_size=32, verbose=0)
	print("%s:\t%.4f" % (model.metrics_names[0], scores[0]))
	print("%s:\t%.2f%%" % (model.metrics_names[1], scores[1]*100))


"""Testing"""
if __name__ == '__main__':
    try:
        arg1 = sys.argv[1]
    except IndexError:
        print("Command line argument missing. Usage: load_binary_nets.py <net number>")
        sys.exit(1)
    if arg1 == '1':
    	load_binary_net1()
    elif arg1 == '2':
    	load_binary_net2()
    elif arg1 == '3':
    	load_binary_net3()
    else:
    	print("Wrong command line argument. Must be a value between 1-3.")


