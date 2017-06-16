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
	return_values.append(corners.shape[0])
	print(type(windows))
	print(windows.shape)
	print("Data loaded")
	windows, corners, scores = predict_binary_net1(windows, corners)
	return_values.append(corners.shape[0])
	np.save(PATH + 'Results/corners_net1_prediction_' + image_name + '.npy',corners)
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
	np.save(PATH + 'Results/corners_net1_calibration_' + image_name + '.npy',corners)
	corners = non_max_suppression_slow(corners, OVERLAPPING_THRES, scores)
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
	windows, corners, scores = predict_binary_net2(windows2, windows1, corners)
	return_values.append(corners.shape[0])
	np.save(PATH + 'Results/corners_net2_prediction_' + image_name + '.npy',corners)
	dispWindows(image, corners, disp)
	labels = predict_calib_net2(windows)
	print("predict_calib_net2")
	corners = calibrate(corners, labels, movsDict)
	dispWindows(image, corners, disp)
	return_values.append(corners.shape[0])
	print("number of corners after net 2:", corners.shape[0])
	np.save(PATH + 'Results/corners_net2_calibration_' + image_name + '.npy',corners)
	corners = non_max_suppression_slow(corners, OVERLAPPING_THRES, scores)
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
	windows, corners, scores = predict_binary_net3(windows3, windows2, windows1, corners)
	return_values.append(corners.shape[0])
	np.save(PATH + 'Results/corners_net3_prediction_' + image_name + '.npy',corners)
	dispWindows(image, corners, disp)
	labels = predict_calib_net3(windows)
	print("predict_calib_net3")
	corners = calibrate(corners, labels, movsDict)
	dispWindows(image, corners, disp)
	return_values.append(corners.shape[0])
	print("number of corners after net 3:", corners.shape[0])
	np.save(PATH + 'Results/corners_net3_calibration_' + image_name + '.npy',corners)
	corners = non_max_suppression_slow(corners, OVERLAPPING_THRES, scores)
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
	avg_recall = [0.0]*9
	avg_precision = [0.0]*9
	num_files = 0.0

	for image_name in file_names:
		if image_name.endswith('.jpg'):
			num_files += 1
			image = Image.open(PATH + path + image_name)
			imageDotted = Image.open(PATH + pathDotted + image_name)

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

			corners = [None] * 9
			corners[0] = np.load(PATH + 'Results/corners_net1_prediction_' + image_name + '.npy')
			corners[1] = np.load(PATH + 'Results/corners_net1_calibration_' + image_name + '.npy')
			corners[2] = np.load(PATH + 'Results/corners_net1_' + image_name + '.npy')
			corners[3] = np.load(PATH + 'Results/corners_net2_prediction_' + image_name + '.npy')
			corners[4] = np.load(PATH + 'Results/corners_net2_calibration_' + image_name + '.npy')
			corners[5] = np.load(PATH + 'Results/corners_net2_' + image_name + '.npy')
			corners[6] = np.load(PATH + 'Results/corners_net3_prediction_' + image_name + '.npy')
			corners[7] = np.load(PATH + 'Results/corners_net3_calibration_' + image_name + '.npy')
			corners[8] = np.load(PATH + 'Results/corners_net3_' + image_name + '.npy')
			
			for stage in range(len(corners)):
				print("Found ", len(corners[stage]), " lions")
				precision = 0.0 # sealions in our res / total res
				recall = 0.0 # sealions in our res / total sealions
				count = 0.0
				total_dots = 0.0

				for lion_class in classes:
					for coordinate in coordinates[lion_class][image_name]:
						#print(coordinate[0]-ORIGINAL_WINDOW_DIM/2, coordinate[1]-ORIGINAL_WINDOW_DIM/2)
						total_dots += 1
						for corner in corners[stage]:
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
				try: 
					recall = count / total_dots
					precision = count / len(corners[stage])
					print("recall: ", recall)
					print("precision: ", precision)
					avg_recall[stage] += recall
					avg_precision[stage] += precision
				except:
					print("Skipping image ", image_name, " as it contains no sealions.")
					num_files -= 1

	for stage in range(len(avg_recall)):
		avg_recall[stage] /= num_files
		avg_precision[stage] /= num_files

	print("AVG recall: ", avg_recall)
	print("AVG precision: ", avg_precision)


def test_folder(path, pathResults, pathDotted=None):
	file_names = os.listdir(PATH + path)
	# Create list of image names that have been already tested
	result_file_names = os.listdir(PATH + pathResults)
	result_file_names = [name for name in result_file_names if name.startswith("corners_net3_")]
	result_file_names = [name[13:-4] for name in result_file_names]
	result_file_names = set(result_file_names)
	# Average number of windows initially and after each of 9 stages (3 x binary/calibr/nms)
	avg_windows = [0.0]*10
	num_files = 0.0
	for image_name in file_names:
		if image_name.endswith('.jpg') and image_name not in result_file_names:
			num_files += 1
			image = Image.open(PATH + path + image_name)
			print(image_name)
			if (pathDotted):
				imageDotted = Image.open(PATH + pathDotted + image_name)
				windows = test_net(image, image_name, imageDotted)
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
		print("Command line argument missing. Usage: test_pipeline.py path_to_test/ path_to_results/ (e.g. Data/Test/ Results/)")
		sys.exit(1)

	if (len(sys.argv)==5):
		test_folder(arg1, sys.argv[2], sys.argv[3])
		evaluate_result(arg1, sys.argv[3], visualize=True)
	elif (len(sys.argv)==4):
		test_folder(arg1, sys.argv[2], sys.argv[3])
		evaluate_result(arg1, sys.argv[3])
	else:
		test_folder(arg1, sys.argv[2])
