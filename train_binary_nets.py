import pdb
import sys
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import ModelCheckpoint
from PIL import Image
import matplotlib.pyplot as plt
from binary_nets import *

def train_net1():

	# Instante HDF5Matrix for the training set
	X_train = HDF5Matrix('data_net3_small.h5', 'data', start=0, end=250)
	y_train = HDF5Matrix('data_net3_small.h5', 'labels', start=0, end=250)
	print X_train.shape
	print y_train.shape

	# # Check some data
	# for idx in range(50):
	# 	print y_train[idx]
	# 	plt.imshow(X_train[idx])
	# 	plt.show()

	# Instante HDF5Matrix for the test set
	X_test = HDF5Matrix('data_net3_small.h5', 'data', start=250, end=300)
	y_test = HDF5Matrix('data_net3_small.h5', 'labels', start=250, end=300)
	print X_test.shape
	print y_test.shape

	# Zero center
	means = np.mean(X_train, axis = 0)
	X_train -= means
	X_test -= means


	layer, model = build_net_1(Input(shape=(100, 100, 3)))

	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

	# THIS IS NOT WORKING ATM (COMMENT OUT callbacks in model.fit to make it run)
 	#checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
 	checkpointer = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, 
 		save_weights_only=False, mode='auto', period=1)

	print model.summary()
	# Note: you have to use shuffle='batch' or False with HDF5Matrix
	#model.fit(X_train, y_train, batch_size=32, shuffle='batch')

	model.fit(X_train, y_train,
	          batch_size=32,
	          epochs=30,
	          validation_data=(X_test, y_test),
	          shuffle='batch',
	          verbose=2, 
	        	callbacks=[checkpointer])

	model.evaluate(X_test, y_test, batch_size=32)


def train_net2():
	pass

def train_net3():
	pass


if __name__ == '__main__':
    try:
        arg1 = sys.argv[1]
    except IndexError:
        print("Command line argument missing. Usage: train_net.py <net number>")
        sys.exit(1)
    if arg1 == '1':
    	train_net1()
    elif arg1 == '2':
    	train_net2()
    elif arg1 == '3':
    	train_net3()
    else:
    	print("Wrong command line argument. Must be a value between 1-3.")


