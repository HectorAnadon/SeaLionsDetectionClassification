import pdb
import sys
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import ModelCheckpoint, Callback
from PIL import Image
import matplotlib.pyplot as plt
from binary_nets import *
from global_variables import *
from make_datasets import *


def train_binary_net1():
	class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.metrics = []

        def on_epoch_end(self, epoch, logs={}):
            self.metrics.append(logs)
            np.save(PATH + 'Results/loss_binary_net1.npy', np.array(self.metrics))

	"""Train the first binary net and save training data means and best model weights.
	"""
	# Load data
	X_data = HDF5Matrix(PATH + 'Datasets/data_net1_small.h5', 'data')
	y_data = HDF5Matrix(PATH + 'Datasets/data_net1_small.h5', 'labels')
	# Split into training and validation sets
	X_train, y_train, X_test, y_test = split_data(X_data, y_data, TRAIN_SPLIT)
	# Zero center
	means = np.mean(X_train, axis = 0)
	X_train -= means
	X_test -= means
	# Save means (for testing)
	np.save(PATH + 'Datasets/means_net1.npy',means)
	# Create model
	layer, model = build_net_1(Input(shape=(X_train.shape[1], X_train.shape[2], 3)))
	print(model.summary())
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
	history = LossHistory()
 	# Checkpoint (for saving the weights)
	filepath = PATH + 'Weights/weights_binary_net1.hdf5'
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, 
			save_weights_only=True, mode='max')
	callbacks_list = [checkpoint, history]
	# Train model (and save the weights)
	model.fit(X_train, y_train,
			batch_size=32,
			epochs=30,
			validation_data=(X_test, y_test),
			shuffle='batch', # Have to use shuffle='batch' or False with HDF5Matrix
			verbose=0, 
			callbacks=callbacks_list)


def train_binary_net2():
	class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.metrics = []

        def on_epoch_end(self, epoch, logs={}):
            self.metrics.append(logs)
            np.save(PATH + 'Results/loss_binary_net2.npy', np.array(self.metrics))

	"""Train the second binary net and save training data means and best model weights.
	"""
	# Load data (current net)
	X_data = HDF5Matrix(PATH + 'Datasets/data_net2_small.h5', 'data')
	y_data = HDF5Matrix(PATH + 'Datasets/data_net2_small.h5', 'labels')
	# Split into training and validation sets
	X_train, y_train, X_test, y_test = split_data(X_data, y_data, TRAIN_SPLIT)
	# Zero center
	means = np.mean(X_train, axis = 0)
	X_train -= means
	X_test -= means
	# Save means (for testing)
	np.save(PATH + 'Datasets/means_net2.npy', means)
	# Load data (previous net)
	X_data = HDF5Matrix(PATH + 'Datasets/data_net1_small.h5', 'data')
	y_data = HDF5Matrix(PATH + 'Datasets/data_net1_small.h5', 'labels')
	# Split into training and validation sets
	X_train_prev, y_train_prev, X_test_prev, y_test_prev = split_data(X_data, y_data, TRAIN_SPLIT)
	# Zero center
	means = np.load(PATH + 'Datasets/means_net1.npy')
	X_train_prev -= means
	X_test_prev -= means

	# Check the labels are the same
	assert np.array_equal(y_train, y_train_prev) and np.array_equal(y_test, y_test_prev)

	# Create model
	print(X_train.shape[1], X_train.shape[2])
	layer, model = build_net_2(Input(shape=(X_train.shape[1], X_train.shape[2], 3)))
	print(model.summary())
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
	history = LossHistory()
 	# Checkpoint (for saving the weights)
	filepath = PATH + 'Weights/weights_binary_net2.hdf5'
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, 
			save_weights_only=True, mode='max')
	callbacks_list = [checkpoint, history]
	# Train model (and save the weights)
	model.fit([X_train, X_train_prev], y_train,
			batch_size=32,
			epochs=30,
			validation_data=([X_test, X_test_prev], y_test),
			shuffle='batch', # Have to use shuffle='batch' or False with HDF5Matrix
			verbose=0, 
			callbacks=callbacks_list)


def train_binary_net3():
	class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.metrics = []

        def on_epoch_end(self, epoch, logs={}):
            self.metrics.append(logs)
            np.save(PATH + 'Results/loss_binary_net3.npy', np.array(self.metrics))

	"""Train the third binary net and save training data means and best model weights.
	"""
	# Load data (current net)
	X_data = HDF5Matrix(PATH + 'Datasets/data_net3_small.h5', 'data')
	y_data = HDF5Matrix(PATH + 'Datasets/data_net3_small.h5', 'labels')
	# Split into training and validation sets
	X_train, y_train, X_test, y_test = split_data(X_data, y_data, TRAIN_SPLIT)
	# Zero center
	means = np.mean(X_train, axis = 0)
	X_train -= means
	X_test -= means
	# Save means (for testing)
	np.save(PATH + 'Datasets/means_net3.npy', means)
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
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
	history = LossHistory()
 	# Checkpoint (for saving the weights)
	filepath = PATH + 'Weights/weights_binary_net3.hdf5'
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, 
			save_weights_only=True, mode='max')
	callbacks_list = [checkpoint, history]
	# Train model (and save the weights)
	model.fit([X_train, X_train2, X_train1], y_train,
			batch_size=32,
			epochs=30,
			validation_data=([X_test, X_test2, X_test1], y_test),
			shuffle='batch', # Have to use shuffle='batch' or False with HDF5Matrix
			verbose=0, 
			callbacks=callbacks_list)


"""Testing"""
if __name__ == '__main__':
    try:
        arg1 = sys.argv[1]
    except IndexError:
        print("Command line argument missing. Usage: train_binary_nets.py <net number>")
        sys.exit(1)
    if arg1 == '1':
    	train_binary_net1()
    elif arg1 == '2':
    	train_binary_net2()
    elif arg1 == '3':
    	train_binary_net3()
    else:
    	print("Wrong command line argument. Must be a value between 1-3.")


