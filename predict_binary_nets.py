import pdb
import sys
import matplotlib.pyplot as plt
from keras.utils.io_utils import HDF5Matrix
from binary_nets import *


def predict_net1(X_test, corners_test):
	# Load training data mean 
	# means = HDF5Matrix('means_net1.h5', 'data')
	# Zero center
	# X_test -= means

	# Create model
	layer, model = build_net_1(Input(shape=(X_test.shape[1], X_test.shape[2], 3)))

	# Load weights
	model.load_weights('Weights/weights_net1_best.hdf5')

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





def predict_net2():
	pass


def predict_net3():
	pass


if __name__ == '__main__':

	X_train = HDF5Matrix('data_net1_small.h5', 'data', start=0, end=250)
	output = predict_net1(X_train)
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


