import pdb
import sys
import matplotlib.pyplot as plt
from keras.utils.io_utils import HDF5Matrix
from classification_net import *
from global_variables import *
import numpy as np

def predict_classification_net(X_test, image_name):
	""" Predict labels for binary net 1 and return only windows containing sealions.
	"""
	# Load training data mean 
	means = np.load(PATH + 'Datasets/means_classification.npy')
	# Zero center
	X_test -= means
	# Create model
	model = classification_net()
	# Load weights
	model.load_weights(PATH + 'Weights/weights_classification_net.hdf5')
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
	# Predict model
	scores = model.predict(X_test)
	# Get indexes of the windows labeled as sealions (0 because sealions are [1 0])
	prediction = np.argmax(scores, axis=1)

	np.save(PATH + 'Results/classification_'+ image_name + '.npy', prediction)
	return prediction



"""Testing"""
if __name__ == '__main__':

	X = HDF5Matrix(PATH + 'Datasets/data_positive_net4_small.h5', 'data', start=500, end=550)
	labels = HDF5Matrix(PATH + 'Datasets/data_positive_net4_small.h5', 'labels', start=500, end=550)
	output = predict_classification_net(X, 'dummy')

	print (output.shape)
	print (labels.shape)

	for i in range(output.shape[0]):
		print(output[i], np.argmax(labels[i]))