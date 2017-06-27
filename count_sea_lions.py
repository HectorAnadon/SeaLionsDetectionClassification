import numpy as np
from collections import Counter
from global_variables import *
from test_functions import *
import os, sys
from PIL import Image
from predict_classification_net import predict_classification_net
from usefulFunctions import extractCoordinates

def count_sea_lions(image_name, image, brief=None):
	if (brief):
		corners = np.load(PATH + 'Results/corners_net3_' + image_name + '.npy')
		windows = getWindows(corners, image, 1)
		prediction = predict_classification_net(windows, image_name)
		counts = Counter(prediction) 
		counts = np.array([counts[idx] for idx in range(len(CLASSES))], dtype=np.int)
		np.save(PATH + 'Results/count_'+ image_name + '.npy', counts)
		print('created counts for image ', image_name, counts)
		return counts
	corners = np.load(PATH + 'Results/corners_net3_' + image_name + '.npy')
	windows = getWindows(corners, image, 1)
	prediction = predict_classification_net(windows, image_name)
	counts = Counter(prediction) 
	np.save(PATH + 'Results/count_'+ image_name + '.npy', counts)
	print(image_name, counts)
	return corners, prediction, counts


def count_folder(path):
	# Submission numpy array
	num_test_files = 18636
	submission = np.zeros((num_test_files, len(CLASSES) + 1), dtype=np.int)
	submission[:,0] = np.array(range(num_test_files))
	file_header = 'test_id,'+','.join(CLASSES)
	# Count sea lions and update corresponding row in the array
	file_names = os.listdir(PATH + path)
	c = 0
	for image_name in file_names:
		if image_name.endswith('.jpg'):
			c += 1
			image = Image.open(PATH + path + image_name)
			try: 
				row = np.load(PATH + 'Results/count_'+ image_name + '.npy')
				submission[int(image_name[:-4]), 1:] = row
				print('loaded counts for image ', image_name, row)
			except:
				try:
					counts = count_sea_lions(image_name, image, True)
					row = np.array([counts[idx] for idx in range(len(CLASSES))], dtype=np.int)
					submission[int(image_name[:-4]), 1:] = row
				except:
					print('Skipping image ', image_name)
			if c%100 == 0:
				# Save to csv (intermediary)
				print("SAVE (c = ", c, ")")
				np.savetxt(PATH+"Results/intermediary_submission.csv", submission, fmt='%d', 
					delimiter=',', newline='\n', header=file_header, comments='')
	# Save to csv (final)
	np.savetxt(PATH+"Results/final_submission.csv", submission, fmt='%d', delimiter=',', newline='\n', 
		header=file_header, comments='')


def create_unify_csv(path):
	# Submission numpy array
	start_file = 1
	num_test_files = 18636
	submission = np.zeros((num_test_files, len(CLASSES) + 1), dtype=np.int)
	submission[:,0] = np.array(range(num_test_files))
	file_header = 'test_id,'+','.join(CLASSES)

	# Count sea lions and update corresponding row in the array
	while(start_file <= num_test_files):
		try: 
			row = np.load(PATH + path + 'count_'+ str(start_file) + '.jpg.npy')
			submission[start_file, 1:] = row
			print('loaded counts for image ', start_file, row)
		except:
			print('empty', start_file)

		start_file += 1


	# Save to csv (final)
	np.savetxt(PATH+"Results/final_submission.csv", submission, fmt='%d', delimiter=',', newline='\n', 
		header=file_header, comments='')




def evaluate_result(path, pathDotted, visualize=False):
	file_names = os.listdir(PATH + path)
	avg_recall = [0.0]*5
	avg_precision = [0.0]*5
	num_files = [0.0]*5
	num_files_global = 0.0
	rms_global = 0.0

	for image_name in file_names:
		if image_name.endswith('.jpg'):
			num_files_global += 1
			rms = [0.0] * 5
			for i in range(len(num_files)):
				num_files[i] += 1
			image = Image.open(PATH + path + image_name)
			imageDotted = Image.open(PATH + pathDotted + image_name)

			parent_folder = path.split("/")
			parent = ""
			for i in range(len(parent_folder)-2):
				parent += parent_folder[i]
				parent += "/"
			coordinates = extractCoordinates(PATH, image_name, parent)

			corners, prediction, counts = count_sea_lions(image_name, image)

			classes = enumerate(CLASSES)
			for class_index, lion_class in classes:
				precision = 0.0 # sealions in our res / total res
				recall = 0.0 # sealions in our res / total sealions
				count = 0.0
				total_dots = 0.0
				for coordinate in coordinates[lion_class][image_name]:
					#print(coordinate[0]-ORIGINAL_WINDOW_DIM/2, coordinate[1]-ORIGINAL_WINDOW_DIM/2)
					total_dots += 1
					for idx in range(len(corners)):
						corner = corners[idx]
						if prediction[idx] == class_index and abs(corner[0] - (coordinate[0]-ORIGINAL_WINDOW_DIM/2)) < EVALUATION_MARGIN and \
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
					print("class: ", lion_class)
					print("count: ", count) 
					print("total_dots", total_dots)
					print("our estim", counts[class_index])
					recall = count / total_dots
					precision = count / counts[class_index]
					print("recall: ", recall)
					print("precision: ", precision)
					avg_recall[class_index] += recall
					avg_precision[class_index] += precision
				except:
					print("Skipping image ", image_name, " as it contains no ", lion_class)
					num_files[class_index] -= 1

				rms[class_index] = (count - total_dots)**2.0

			rms_value = 0
			for i in range(len(rms)):
				rms_value += rms[i]
			rms_value /= 5.0
			rms_value = np.sqrt(rms_value)

			rms_global += rms_value



	for idx in range(len(avg_recall)):
		avg_recall[idx] /= num_files[idx]
		avg_precision[idx] /= num_files[idx]
	rms_global /= num_files_global

	print("AVG recall: ", avg_recall)
	print("AVG precision: ", avg_precision)
	print ("RMS: ", rms_global)



        	

"""Testing"""
if __name__ == '__main__':

	try:
		arg1 = sys.argv[1]
	except IndexError:
		print("Command line argument missing. Usage: count_sea_lions.py path_to_folder/ (Test/)")
		sys.exit(1)

	if (len(sys.argv)==4):
		evaluate_result(arg1, sys.argv[2], visualize=True)
	elif (len(sys.argv)==3):
		if (sys.argv[2] == 'csv'):
			create_unify_csv(arg1)
		else:
			evaluate_result(arg1, sys.argv[2])
	else:
		count_folder(arg1)



