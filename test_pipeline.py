import numpy as np
from predict_binary_nets import *
from predict_binary_nets import *
from test_functions import *
from usefulFunctions import *
import os, sys
from PIL import Image
import matplotlib.pyplot as plt
from global_variables import *
from predict_calibration_nets import *


def test_net(image, image_name, imageDotted=None, disp=False):
	# NET 1
	if (imageDotted):
		windows, corners = sliding_window_net_1(image, imageDotted)
	else:
		windows, corners = sliding_window_net_1(image)
	return_values = []
	print(type(windows))
	print(windows.shape)
	print("Data loaded")
	windows, corners = predict_binary_net1(windows, corners)
	dispWindows(image, corners, disp)
	print("Predict_binary_net1")
	print(windows.shape)
	labels = predict_calib_net1(windows)
	print("predict_calib_net1")
	movsDict = createCalibrationDictionary()
	corners = calibrate(corners, labels, movsDict)
	dispWindows(image, corners, disp)
	return_values.append(corners.shape[0])
	print("number of corners after net 1:", corners.shape[0])
	corners = non_max_suppression_slow(corners, OVERLAPPING_THRES)
	return_values.append(corners.shape[0])
	print("number of corners after NMS:", corners.shape[0])
	np.save(PATH + 'Results/corners_net1_' + image_name + '.npy',corners)
	dispWindows(image, corners, disp)
	# for corner in corners:
	# 	plt.imshow(cropAndChangeResolution(image,image_name,corner[0],corner[1],ORIGINAL_WINDOW_DIM,1))
	# 	plt.show()

	# NET 2
	# Windows for net 2
	windows2 = getWindows(corners, image, 2)
	windows1 = getWindows(corners, image, 3)
	print(windows1.shape)
	windows, corners = predict_binary_net2(windows2, windows1, corners)
	dispWindows(image, corners, disp)
	labels = predict_calib_net2(windows)
	print("predict_calib_net2")
	corners = calibrate(corners, labels, movsDict)
	dispWindows(image, corners, disp)
	return_values.append(corners.shape[0])
	print("number of corners after net 2:", corners.shape[0])
	corners = non_max_suppression_slow(corners, OVERLAPPING_THRES)
	return_values.append(corners.shape[0])
	print("number of corners after NMS:", corners.shape[0])
	np.save(PATH + 'Results/corners_net2_' + image_name + '.npy',corners)
	dispWindows(image, corners, disp)
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
	dispWindows(image, corners, disp)
	labels = predict_calib_net3(windows)
	print("predict_calib_net3")
	corners = calibrate(corners, labels, movsDict)
	dispWindows(image, corners, disp)
	return_values.append(corners.shape[0])
	print("number of corners after net 3:", corners.shape[0])
	corners = non_max_suppression_slow(corners, OVERLAPPING_THRES)
	return_values.append(corners.shape[0])
	print("number of corners after NMS:", corners.shape[0])
	np.save(PATH + 'Results/corners_net3_' + image_name + '.npy',corners)
	dispWindows(image, corners, disp)
	# for corner in corners:
	# 	plt.imshow(cropAndChangeResolution(image,image_name,corner[0],corner[1],ORIGINAL_WINDOW_DIM,1))
	# 	plt.show()
	return return_values



def evaluate_result(path, pathDotted, visualize=False):
	file_names = os.listdir(PATH + path)
	avg_recall = 0.0
	avg_precision = 0.0
	num_files = 0.0

	for image_name in file_names:
		if image_name.endswith('.jpg'):
			num_files += 1
			image = Image.open(PATH + path + image_name)
			#print(sizeInfo(image, resolutions=[1]))

			imageDotted = Image.open(PATH + pathDotted + image_name)

			corners = np.load(PATH + 'Results/corners_net3_' + image_name + '.npy')
			print("Found ", len(corners), " lions")

			parent_folder = path.split("/")
			parent = ""
			for i in range(len(parent_folder)-2):
				parent += parent_folder[i]
				parent += "/"
			coordinates = extractCoordinates(PATH, image_name, parent)

			classes = CLASSES
			s = 0
			for lion_class in CLASSES:
				s = s + len(coordinates[lion_class][image_name])

			print("Extracted ", s, " lions")

			precision = 0.0 # sealions in our res / total res
			recall = 0.0 # sealions in our res / total sealions
			count = 0.0
			total_dots = 0.0

			for lion_class in classes:
				for coordinate in coordinates[lion_class][image_name]:
					#print(coordinate[0]-ORIGINAL_WINDOW_DIM/2, coordinate[1]-ORIGINAL_WINDOW_DIM/2)
					total_dots += 1
					for corner in corners:
						if abs(corner[0] - (coordinate[0]-ORIGINAL_WINDOW_DIM/2)) < EVALUATION_MARGIN and \
								abs(corner[1] - (coordinate[1]-ORIGINAL_WINDOW_DIM/2)) < EVALUATION_MARGIN:
							count += 1
							if visualize:
								print("MATCH: ", corner)
								plt.imshow(cropAndChangeResolution(imageDotted,image_name,corner[0],corner[1],ORIGINAL_WINDOW_DIM,1))
								plt.show()
								plt.imshow(cropAndChangeResolution(image,image_name,corner[0],corner[1],ORIGINAL_WINDOW_DIM,1))
								plt.show()
							break;
					if visualize:
						plt.imshow(cropAndChangeResolution(imageDotted,image_name,coordinate[0]-ORIGINAL_WINDOW_DIM/2,coordinate[1]-ORIGINAL_WINDOW_DIM/2,ORIGINAL_WINDOW_DIM,1))
						plt.show()
			try: # TO DO DECIDE HOW TO DO THIS - TOGETHER WITH THE GUYS!
				recall = count / total_dots
				precision = count / len(corners)
				print("recall: ", recall)
				print("precision: ", precision)
				avg_recall += recall
				avg_precision += precision
			except:
				print("Skipping image ", image_name, " as it contains no sealions.")

	avg_recall /= num_files
	avg_precision /= num_files

	print("AVG recall: ", avg_recall)
	print("AVG precision: ", avg_precision)


def test_folder(path, pathDotted=None):
	file_names = os.listdir(PATH + path)
	avg_windows = [0.0]*6
	num_files = 0.0
	for image_name in file_names:
		if image_name.endswith('.jpg'):
			num_files += 1
			image = Image.open(PATH + path + image_name)
			print(image_name)
			if (pathDotted):
				imageDotted = Image.open(PATH + pathDotted + image_name)
				#windows = test_net(image, image_name, imageDotted)
				windows = [0.2]*6
				print("windows: ", windows)
				for i in range(len(avg_windows)):
					avg_windows[i] += windows[i]
			else:
				test_net(image, image_name)
	for i in range(len(avg_windows)):
		avg_windows[i] /= num_files
	print("AVG windows: ", avg_windows)

if __name__ == '__main__':
	try:
		arg1 = sys.argv[1]
	except IndexError:
		print("Command line argument missing. Usage: test_pipeline.py path_to_folder/ (Test/)")
		sys.exit(1)

	if (len(sys.argv)==4):
		test_folder(arg1, sys.argv[2])
		evaluate_result(arg1, sys.argv[2], visualize=True)
	elif (len(sys.argv)==3):
		test_folder(arg1, sys.argv[2])
		evaluate_result(arg1, sys.argv[2])
	else:
		test_folder(arg1)
