from keras.utils.io_utils import HDF5Matrix
from predict_binary_nets import *
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt


def getWrongWindowsNet1():
	X_data = HDF5Matrix(PATH + 'Datasets/data_net1_small.h5', 'data')
	y_data = HDF5Matrix(PATH + 'Datasets/data_net1_small.h5', 'labels')
	dummy = np.zeros(X_data.shape)

	argmax = predict_binary_net1(X_data, dummy, onlyPrediction=True)
	labels = np.argmax(y_data, axis=1)

	# PRINT false positive and true negative
	#  0 - correct
	#  1 - Sea lion classify as rock
	# -1 - Rock classify as sea lion
	accuracy = argmax - labels
	counts = Counter(accuracy)
	print(counts)

	indexes = []
	for i in range(len(labels)):
		if (argmax[i] != 1 or labels[i] != 1):
			indexes.append(i)

	print(len(indexes))
	np.save(PATH + 'Datasets/indexes_net2.npy', np.array(indexes))


	# new_data = np.take(X_data, indexes)


def getWrongWindowsNet2():
	indexes = np.load(PATH + 'Datasets/indexes_net2.npy')
	print(len(indexes))

	X_data_2 = HDF5Matrix(PATH + 'Datasets/data_net2_small.h5', 'data')
	X_data_2 = X_data_2[indexes]
	X_data_1 = HDF5Matrix(PATH + 'Datasets/data_net1_small.h5', 'data')
	X_data_1 = X_data_1[indexes]
	y_data = HDF5Matrix(PATH + 'Datasets/data_net1_small.h5', 'labels')
	y_data = y_data[indexes]
	dummy = np.zeros(X_data_2.shape)

	argmax = predict_binary_net2(X_data_2, X_data_1, dummy, onlyPrediction=True)
	labels = np.argmax(y_data, axis=1)

	# PRINT false positive and true negative
	#  0 - correct
	#  1 - Sea lion classify as rock
	# -1 - Rock classify as sea lion
	accuracy = argmax - labels
	counts = Counter(accuracy)
	print(counts)

	indexes = []
	for i in range(len(labels)):
		if (argmax[i] != 1 or labels[i] != 1):
			indexes.append(i)

	print(len(indexes))
	np.save(PATH + 'Datasets/indexes_net3.npy', np.array(indexes))


def getStatsNet3():
	indexes = np.load(PATH + 'Datasets/indexes_net3.npy')

	X_data_3 = HDF5Matrix(PATH + 'Datasets/data_net3_small.h5', 'data')
	X_data_3 = X_data_3[indexes]
	X_data_2 = HDF5Matrix(PATH + 'Datasets/data_net2_small.h5', 'data')
	X_data_2 = X_data_2[indexes]
	X_data_1 = HDF5Matrix(PATH + 'Datasets/data_net1_small.h5', 'data')
	X_data_1 = X_data_1[indexes]
	y_data = HDF5Matrix(PATH + 'Datasets/data_net1_small.h5', 'labels')
	y_data = y_data[indexes]
	dummy = np.zeros(X_data_2.shape)

	argmax = predict_binary_net3(X_data_3, X_data_2, X_data_1, dummy, onlyPrediction=True)
	labels = np.argmax(y_data, axis=1)

	# PRINT false positive and true negative
	#  0 - correct
	#  1 - Sea lion classify as rock
	# -1 - Rock classify as sea lion
	accuracy = argmax - labels
	counts = Counter(accuracy)
	print(counts)


"""Testing"""
if __name__ == '__main__':

	
	try:
		arg1 = sys.argv[1]
	except IndexError:
		print("Command line argument missing. Usage: getWrongWindows.py <net number>")
		sys.exit(1)

	if arg1 == '1':
		getWrongWindowsNet1()
	elif arg1 == '2':
		getWrongWindowsNet2()
	elif arg1 == '3':
		getStatsNet3()
	else:
		print("Wrong command line argument. Must be a value between 1-3.")






# FOR OPTION TWO
	# for lion_class in classes:
	# 				for coordinate in coordinates[lion_class][image_name]:
	# 					#print(coordinate[0]-ORIGINAL_WINDOW_DIM/2, coordinate[1]-ORIGINAL_WINDOW_DIM/2)
	# 					total_dots += 1
	# 					for corner in corners[stage]:
	# 						if abs(corner[0] - (coordinate[0]-ORIGINAL_WINDOW_DIM/2)) < EVALUATION_MARGIN and \
	# 								abs(corner[1] - (coordinate[1]-ORIGINAL_WINDOW_DIM/2)) < EVALUATION_MARGIN:
	# 							count += 1
	# 							if visualize:
	# 								print("MATCH: ", corner)
	# 								plt.imshow(cropAndChangeResolution(imageDotted,image_name,corner[0],corner[1],ORIGINAL_WINDOW_DIM,1))
	# 								plt.show()
	# 								plt.imshow(cropAndChangeResolution(image,image_name,corner[0],corner[1],ORIGINAL_WINDOW_DIM,1))
	# 								plt.show()