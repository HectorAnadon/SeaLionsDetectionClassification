import pdb
import sys
import matplotlib.pyplot as plt
from keras.utils.io_utils import HDF5Matrix
from binary_nets import *
from global_variables import*

def predict_binary_net1(X_test, corners_test):
	""" Predict labels for binary net 1 and return only windows containing sealions.
	"""
	# Load training data mean 
	means = np.load(PATH + 'Datasets/means_net1.npy')
	# Zero center
	X_test -= means
	# Create model
	layer, model = build_net_1(Input(shape=(X_test.shape[1], X_test.shape[2], 3)))
	# Load weights
	model.load_weights(PATH + 'Weights/weights_binary_net1.hdf5')
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
	# Predict model
	scores = model.predict(X_test)
	scores[:,0] = np.where(scores[:,0] > ARGMAX_THRESHOLD, scores[:,0], -1)
	# Get indexes of the windows labeled as sealions (0 because sealions are [1 0])
	argmax = np.argmax(scores, axis=1)
	idx = np.argwhere(argmax == 0)
	idx = np.reshape(idx, (idx.shape[0],))
	# Return only windows containing sealions
	return X_test[idx], corners_test[idx], scores[idx,0]


def predict_binary_net2(X_test, X_test_prev, corners_test):
	""" Predict labels for binary net 2 and return only windows containing sealions.
	"""
	# Load training data mean (current net)
	means = np.load(PATH + 'Datasets/means_net2.npy')
	# Zero center
	X_test -= means
	# Load training data mean (previous net)
	means = np.load(PATH + 'Datasets/means_net1.npy')
	# Zero center
	X_test_prev -= means
	# Create model
	layer, model = build_net_2(Input(shape=(X_test.shape[1], X_test.shape[2], 3)))
	# Load weights
	model.load_weights(PATH + 'Weights/weights_binary_net2.hdf5')
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
	# Predict model
	scores = model.predict([X_test, X_test_prev])
	scores[:,0] = np.where(scores[:,0] > ARGMAX_THRESHOLD, scores[:,0], -1)
	# Get indexes of the windows labeled as sealions (0 because sealions are [1 0]) 
	argmax = np.argmax(scores, axis=1)
	idx = np.argwhere(argmax == 0)
	idx = np.reshape(idx, (idx.shape[0],))
	# Return only windows containing sealions
	return X_test[idx], corners_test[idx], scores[idx,0]


def predict_binary_net3(X_test, X_test_2, X_test_1, corners_test):
	""" Predict labels for binary net 3 and return only windows containing sealions.
	"""
	# Load training data mean (current net)
	means = np.load(PATH + 'Datasets/means_net3.npy')
	# Zero center
	X_test -= means
	# Load training data mean (net 2)
	means = np.load(PATH + 'Datasets/means_net2.npy')
	# Zero center
	X_test_2 -= means
	# Load training data mean (net 1)
	means = np.load(PATH + 'Datasets/means_net1.npy')
	# Zero center
	X_test_1 -= means
	# Create model
	model = build_net_3(Input(shape=(X_test.shape[1], X_test.shape[2], 3)))
	# Load weights
	model.load_weights(PATH + 'Weights/weights_binary_net3.hdf5')
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
	# Predict model
	scores = model.predict([X_test, X_test_2, X_test_1])
	scores[:,0] = np.where(scores[:,0] > ARGMAX_THRESHOLD, scores[:,0], -1)
	# Get indexes of the windows labeled as sealions (0 because sealions are [1 0]) 
	argmax = np.argmax(scores, axis=1)
	idx = np.argwhere(argmax == 0)
	idx = np.reshape(idx, (idx.shape[0],))
	# Return only windows containing sealions
	return X_test[idx], corners_test[idx], scores[idx,0]


"""Testing"""
if __name__ == '__main__':

	X1 = HDF5Matrix(PATH + 'Datasets/data_net1_small.h5', 'data', start=0, end=100)
	X2 = HDF5Matrix(PATH + 'Datasets/data_net2_small.h5', 'data', start=0, end=100)
	X3 = HDF5Matrix(PATH + 'Datasets/data_net3_small.h5', 'data', start=0, end=100)
	corners = np.ones((X1.shape[0], 2))
	print(X1.shape, X2.shape, X3.shape, corners.shape)

	try:
		arg1 = sys.argv[1]
	except IndexError:
		print("Command line argument missing. Usage: predict_binary_nets.py <net number>")
		sys.exit(1)

	if arg1 == '1':
		output, corners, scores = predict_binary_net1(X1, corners)
	elif arg1 == '2':
		output, corners, scores = predict_binary_net2(X2, X1, corners)
	elif arg1 == '3':
		output, corners, scores = predict_binary_net3(X3, X2, X1, corners)
	else:
		print("Wrong command line argument. Must be a value between 1-3.")

	print (output.shape)
	print (corners.shape)

	# for i in range(output.shape[0]):
	# 	img = output[i,:,:,:]
	# 	plt.imshow(img)
	# 	plt.show()




