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

def test_net_1(image, image_name, path):
	windows, corners = sliding_window_net_1(image)
	windows, corners = predict_binary_net1(windows, corners)
	labels = predict_calib_net1(windows)
	movsDict = createCalibrationDictionary()
	corners = calibrate(corners, labels, movsDict)
	corners = non_max_suppression_slow(corners, OVERLAPPING_THRES)
	np.save(path + 'Results/corners_net1_' + image_name + '.npy',corners)
	for i in range(corners.shape[0]):
		plt.imshow(cropAndChangeResolution(image,'image_name',corners[i,0],corners[i,1],ORIGINAL_WINDOW_DIM,1)) #TODO: change resolution lvl to 2
		plt.show()
		plt.imshow(windows[i,:,:,:])
		plt.show()

if __name__ == '__main__':
	file_names = os.listdir("Data/Train/")
	image = Image.open("Data/Train/" + file_names[3])
	print(file_names[1])
	test_net_1(image, file_names[1], "")
