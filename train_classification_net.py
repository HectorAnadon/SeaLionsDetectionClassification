import numpy as np
from keras.utils.io_utils import HDF5Matrix
from classification_net import classification_net
from keras.callbacks import ModelCheckpoint
from make_datasets import split_data
from global_variables import *

def train_classification():
	X_data = HDF5Matrix(PATH + 'Datasets/data_positive_net4_small.h5', 'data')
	y_data = HDF5Matrix(PATH + 'Datasets/data_positive_net4_small.h5', 'labels')

	# Split into training and validation sets
	X_train, y_train, X_test, y_test = split_data(X_data, y_data, TRAIN_SPLIT)
	# Zero center
	means = np.mean(X_train, axis = 0)
	X_train -= means
	X_test -= means
	# Save means (for testing)
	np.save(PATH + 'Datasets/means_classification.npy',means)

	model = classification_net()
	model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
	print(model.summary())

	# Checkpoint (for saving the weights)
	filepath = PATH + 'Weights/weights_classification_net.hdf5'
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

if __name__ == '__main__':
	train_classification()