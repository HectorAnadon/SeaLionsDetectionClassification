import numpy as np
from predict_binary_nets import *
from predict_binary_nets import *
from test_functions import *
from usefulFunctions import changeResolution, getWindow
import os
from PIL import Image
import matplotlib.pyplot as plt
from global_variables import *

def test_net_1(image):
	windows, corners = sliding_window_net_1(image)
	windows, corners = predict_net1(windows, corners)
	labels = predict_calib_net1(windows)



if __name__ == '__main__':
	file_names = os.listdir("Data/Train/")
	image = Image.open("Data/Train/" + file_names[0])
	print(file_names[0])
	test_net_1(image)
