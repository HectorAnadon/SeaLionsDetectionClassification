import numpy as np
from predict_binary_nets import *
from predict_binary_nets import *
from test_functions import *
from usefulFunctions import cropAndChangeResolution
import os
from PIL import Image
import matplotlib.pyplot as plt
from global_variables import *
from predict_calibration_nets import *


def test_net(image, image_name, path):
	# NET 1
	windows, corners = sliding_window_net_1(image)
	#windows = HDF5Matrix(path + 'TestDatasets/sliding_window_' + image_name + '.h5', 'data')
	#corners = HDF5Matrix(path + 'TestDatasets/sliding_window_' + image_name + '.h5', 'labels')
	print(type(windows))
	print(windows.shape)
	print("Data loaded")
	windows, corners = predict_binary_net1(windows, corners)
	print("Predict_binary_net1")
	print(windows.shape)
	labels = predict_calib_net1(windows)
	print("predict_calib_net1")
	movsDict = createCalibrationDictionary()
	corners = calibrate(corners, labels, movsDict)
	print("number of corners after net 1:", corners.shape[0])
	corners = non_max_suppression_slow(corners, OVERLAPPING_THRES)
	print("number of corners after NMS:", corners.shape[0])
	np.save(path + 'Results/corners_net1_' + image_name + '.npy',corners)
	# for corner in corners:
	# 	print(100)
	# 	plt.imshow(cropAndChangeResolution(image,image_name,corner[0],corner[1],ORIGINAL_WINDOW_DIM,1))
	# 	plt.show()

	# NET 2
	# Windows for net 2
	windows2 = getWindows(corners, image, 2)
	windows1 = getWindows(corners, image, 3)
	print(windows1.shape)
	windows, corners = predict_binary_net2(windows2, windows1, corners)
	labels = predict_calib_net2(windows)
	print("predict_calib_net2")
	corners = calibrate(corners, labels, movsDict)
	print("number of corners after net 2:", corners.shape[0])
	corners = non_max_suppression_slow(corners, OVERLAPPING_THRES)
	print("number of corners after NMS:", corners.shape[0])
	np.save(path + 'Results/corners_net2_' + image_name + '.npy',corners)
	# for corner in corners:
	# 	plt.imshow(cropAndChangeResolution(image,image_name,corner[0],corner[1],ORIGINAL_WINDOW_DIM,1))
	# 	plt.show()

	# NET 3
	# Windows for net 3
	windows3 = getWindows(corners, image, 1)
	windows2 = getWindows(corners, image, 2)
	windows1 = getWindows(corners, image, 3)
	print(windows1.shape)
	windows, corners = predict_binary_net3(windows3, windows2, windows1, corners)
	labels = predict_calib_net3(windows)
	print("predict_calib_net3")
	corners = calibrate(corners, labels, movsDict)
	print("number of corners after net 3:", corners.shape[0])
	corners = non_max_suppression_slow(corners, OVERLAPPING_THRES)
	print("number of corners after NMS:", corners.shape[0])
	np.save(path + 'Results/corners_net3_' + image_name + '.npy',corners)
	for corner in corners:
		plt.imshow(cropAndChangeResolution(image,image_name,corner[0],corner[1],ORIGINAL_WINDOW_DIM,1))
		plt.show()


if __name__ == '__main__':
	file_names = os.listdir("Data/Train/")
	image = Image.open("Data/Train/" + file_names[0])
	print(file_names[0])
	test_net(image, file_names[0], "")
