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


def test_net_1(image):
	windows, corners = sliding_window_net_1(image)
	windows, corners = predict_net1(windows, corners)
	labels = predict_calib_net1(windows)
	movsDict = createCalibrationDictionary()
	corners = calibrate(corners, labels, movsDict)
	corners = non_max_suppression_slow(corners, OVERLAPPING_THRES)
	for corner in corners:
		plt.imshow(cropAndChangeResolution(image,'image_name',corner[0],corner[1],ORIGINAL_WINDOW_DIM,1)) #TODO: change resolution lvl to 2
		plt.show()


if __name__ == '__main__':
	file_names = os.listdir("Data/Train/")
	image = Image.open("Data/Train/" + file_names[0])
	print(file_names[0])
	test_net_1(image)
