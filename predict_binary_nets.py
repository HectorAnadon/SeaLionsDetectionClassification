import pdb
import sys
import matplotlib.pyplot as plt
from keras.utils.io_utils import HDF5Matrix
from binary_nets import *


def predict_binary_net1(X_test, corners_test):
	""" Predict labels for binary net 1 and return only windows containing sealions.
	"""
	# Load training data mean 
	means = np.load('Datasets/means_net1.npy')
	# Zero center
	X_test -= means
	# Create model
	layer, model = build_net_1(Input(shape=(X_test.shape[1], X_test.shape[2], 3)))
	# Load weights
	model.load_weights('Weights/weights_binary_net1.hdf5')
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
	# Predict model
	scores = model.predict(X_test)
	# Get indexes of the windows labeled as sealions (0 because sealions are [1 0]) 
	argmax = np.argmax(scores, axis=1)
	idx = np.argwhere(argmax == 0)
	idx = np.reshape(idx, (idx.shape[0],))
	# Return only windows containing sealions
	return X_test[idx], corners_test[idx]


# TODO NEED TO BE CAREFUL WITH THE INPUTS HERE! NEED 2 INPUTS -- SEE TRAINING
def predict_binary_net2(X_test, X_test_prev, corners_test):
	""" Predict labels for binary net 2 and return only windows containing sealions.
	"""
	# Load training data mean (current net)
	means = np.load('Datasets/means_net2.npy')
	# Zero center
	X_test -= means
	# Load training data mean (previous net)
	means = np.load('Datasets/means_net2.npy')
	# Zero center
	X_test_prev -= means
	# Create model
	layer, model = build_net_2(Input(shape=(X_test.shape[1], X_test.shape[2], 3)))
	# Load weights
	model.load_weights('Weights/weights_binary_net2.hdf5')
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
	# Predict model
	scores = model.predict(X_test)
	# Get indexes of the windows labeled as sealions (0 because sealions are [1 0]) 
	argmax = np.argmax(scores, axis=1)
	idx = np.argwhere(argmax == 0)
	idx = np.reshape(idx, (idx.shape[0],))
	# Return only windows containing sealions
	return X_test[idx], corners_test[idx]


# TODO NEED TO BE CAREFUL WITH THE INPUTS HERE! NEED 2 INPUTS -- SEE TRAINING
def predict_binary_net3():
	pass


"""Testing"""
if __name__ == '__main__':

	X_train = HDF5Matrix('data_net1_small.h5', 'data', start=0, end=250)
	output = predict_binary_net1(X_train)
	print (output.shape)
	for i in range(output.shape[0]):
		img = output[i,:,:,:]
		plt.imshow(img)
		plt.show()


    # try:
    #     arg1 = sys.argv[1]
    # except IndexError:
    #     print("Command line argument missing. Usage: load_binary_nets.py <net number>")
    #     sys.exit(1)
    # if arg1 == '1':
    # 	load_net1()
    # elif arg1 == '2':
    # 	load_net2()
    # elif arg1 == '3':
    # 	load_net3()
    # else:
    # 	print("Wrong command line argument. Must be a value between 1-3.")


