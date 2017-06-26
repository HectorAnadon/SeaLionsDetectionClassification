import numpy as np
from keras.utils.io_utils import HDF5Matrix
from classification_net import classification_net
from classification_net_simple import build_classif_net
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Input
from make_datasets import split_data
from global_variables import *
from keras.preprocessing.image import ImageDataGenerator

def train_classification():
	class LossHistory(Callback):
		def on_train_begin(self, logs={}):
			self.metrics = []

		def on_epoch_end(self, epoch, logs={}):
			self.metrics.append(logs)
			np.save(PATH + 'Results/loss_classification_net.npy', np.array(self.metrics))

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

	model = build_classif_net() # Simple
	#model = classification_net() # VGG
	model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
	print(model.summary())

	history = LossHistory()

	# Checkpoint (for saving the weights)
	filepath = PATH + 'Weights/weights_classification_net.hdf5'
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, 
			save_weights_only=True, mode='max')
	callbacks_list = [checkpoint, history]

	datagen = ImageDataGenerator(
		featurewise_center=False,
		samplewise_center=False,
		featurewise_std_normalization=False,
		samplewise_std_normalization=False,
		zca_whitening=False,
		rotation_range=50,
		width_shift_range=0.,
		height_shift_range=0.,
		shear_range=0.,
		zoom_range=0.05,
		channel_shift_range=0.,
		fill_mode='nearest',
		cval=0.,
		horizontal_flip=False,
		vertical_flip=True,
		rescale=None,
		preprocessing_function=None)

	model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
						steps_per_epoch=len(X_train) / 32, epochs=100,
						validation_data = (X_test, y_test),
						verbose=1,
						callbacks=callbacks_list)

if __name__ == '__main__':
	train_classification()

