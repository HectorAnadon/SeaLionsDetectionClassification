from keras.utils.io_utils import HDF5Matrix
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import time
import h5py


from usefulFunctions import *
from global_variables import *


def get_positive_samples(path, radius, net):
    """Get an array of positive samples (windows containing lions), their upper left corner 
    coordinates and their labels (both in binary and multiclass one-hot representation)
    """
    resolution_lvl = get_resolution_level(net)
    file_names = os.listdir(path + "Train/")
    positive_samples = []
    # corners = []
    binary_labels = []
    multiclass_labels = []

    for image_name in file_names:
        # Ignore OSX files
        if image_name != ".DS_Store" or image_name != "train.csv":
            print("Processing ", image_name)
            image = Image.open(path + "Train/" + image_name)
            #CHECK IF THERE IS A SEA LION IN THE IMAGE
            coordinates = extractCoordinates(path, image_name)
            classes = enumerate(CLASSES)
            for class_index, class_name in classes:
                for lion in range(len(coordinates[class_name][image_name])):
                    # Only consider sea lions within radius pixels from the edge
                    # TODO consider edge cases
                    x = coordinates[class_name][image_name][lion][0]
                    y = coordinates[class_name][image_name][lion][1]
                    if x > radius and x < (image.size[0] - radius) and y > radius and y < (image.size[1] - radius):
                        # Crop window of chosen resolution level
                        window = cropAndChangeResolution(image, image_name, x - radius, y - radius, radius * 2, resolution_lvl)
                        # Append
                        positive_samples.append(np.array(window))
                        # corners.append(np.array([x - radius, y - radius]))
                        if (net <= 3):
                            binary_labels.append([1,0])
                        else:
                            multiclass =  np.zeros(5, 'uint8')
                            multiclass[class_index] = 1
                            multiclass_labels.append(multiclass)
    # Concatenate
    positive_samples = np.float64(np.stack(positive_samples))
    #corners = np.uint16(np.stack(corners))
    if (net <= 3):
        binary_labels = np.uint8(np.array(binary_labels))
    else:
        multiclass_labels = np.uint8(np.stack(multiclass_labels))
    # Normalize data
    positive_samples /= 255.0
    # Save to disk
    f = h5py.File('data_positive_net'+str(net)+'_small.h5', 'w')
    # Create dataset to store images
    X_dset = f.create_dataset('data', positive_samples.shape, dtype='f')
    X_dset[:] = positive_samples
    print(X_dset.shape)
    # Create dataset to store corners
    # corners_dset = f.create_dataset('corners', corners.shape, dtype='i')
    # corners_dset[:] = corners
    # Create dataset to store labels
    if (net <= 3):
        y_dset = f.create_dataset('labels', binary_labels.shape, dtype='i')
        y_dset[:] = binary_labels
        print (y_dset.shape)
    else:
        y_dset = f.create_dataset('labels', multiclass_labels.shape, dtype='i')
        y_dset[:] = multiclass_labels  
        print (y_dset.shape)
    f.close()
    return positive_samples.shape, binary_labels.shape


def get_negative_samples(path, radius, net):
    """Get an array of negative samples (windows without lions), their upper left corner 
    coordinates and their labels (only binary format - NO SEA LION [0,1] / SEA LION [1,0])
    """
    resolution_lvl = get_resolution_level(net)
    file_names = os.listdir(path + "Train/")
    negative_samples = []
    corners = []
    labels = []

    for image_name in file_names:
        if image_name != ".DS_Store" or image_name != "train.csv":
            print("Processing ", image_name)
            image = Image.open(path + "Train/" + image_name)
            coordinates = extractCoordinates(path, image_name)
            for it in range(NUM_NEG_SAMPLES):
                # Upper left corner coordinates
                x = np.random.uniform(0, image.size[0] - 2 * radius)
                y = np.random.uniform(0, image.size[1] - 2 * radius)
                window = cropAndChangeResolution(image, image_name, x, y, radius * 2, resolution_lvl)
                label = getLabel(image_name, coordinates, x, y, radius * 2)
                # Append negative samples 
                if label == [0, 1]:
                    negative_samples.append(np.array(window))
                    # corners.append(np.array([x, y]))
                    labels.append(label)
    # Concatenate
    negative_samples = np.float64(np.stack(negative_samples))
    # corners = np.uint16(np.stack(corners))
    labels = np.uint8(np.array(labels))
    # Normalize data
    negative_samples /= 255.0
    # Save to disk
    f = h5py.File('data_negative_net'+str(net)+'_small.h5', 'w')
    # Create dataset to store images
    X_dset = f.create_dataset('data', negative_samples.shape, dtype='f')
    X_dset[:] = negative_samples
    print (X_dset.shape)
    # Create dataset to store corners
    # corners_dset = f.create_dataset('corners', corners.shape, dtype='i')
    # corners_dset[:] = corners
    # Create dataset to store labels
    y_dset = f.create_dataset('labels', labels.shape, dtype='i')
    y_dset[:] = labels
    print (y_dset.shape)
    f.close()
    return negative_samples.shape, labels.shape


def unison_shuffled_copies(a, b, c=None, d=None):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    if c != None:
        assert len(a) == len(c)
        if d != None:
            assert len(a) == len(d)
            return a[p], b[p], c[p], d[p]
        else:
            return a[p], b[p], c[p]
    else:
        return a[p], b[p]


def create_net_dataset(path, window_size, net):
    """Combine positive and negative samples into one dataset.
    """
    radius = round(window_size / 2)
    # Load positive samples
    pos_samples = HDF5Matrix('Datasets/data_positive_net'+str(net)+'_small.h5', 'data')
    pos_labels = HDF5Matrix('Datasets/data_positive_net'+str(net)+'_small.h5', 'labels')
    # Load negative samples
    neg_samples = HDF5Matrix('Datasets/data_negative_net'+str(net)+'_small.h5', 'data')
    neg_labels = HDF5Matrix('Datasets/data_negative_net'+str(net)+'_small.h5', 'labels')
    # Check normalization
    assert np.amax(pos_samples) <= 1 and np.amax(neg_samples) <= 1
    # Concatenate positive and negative
    X = np.concatenate((pos_samples, neg_samples))
    y = np.concatenate((pos_labels, neg_labels))
    # Shuffle data
    X, y = unison_shuffled_copies(X, y)
    # Save to disk
    f = h5py.File('Datasets/data_net'+str(net)+'_small.h5', 'w')
    # Create dataset to store images
    X_dset = f.create_dataset('data', X.shape, dtype='f')
    X_dset[:] = X
    # Create dataset to store corners
    # corners_dset = f.create_dataset('corners', corners.shape, dtype='i')
    # corners_dset[:] = corners
    # Create dataset to store labels
    y_dset = f.create_dataset('labels', y.shape, dtype='i')
    y_dset[:] = y
    f.close()
    return X.shape, y.shape


def get_shifted_windows(image, image_name, x, y, resolution_lvl):

    # Offset vectors
    x_n = np.array(X_N)
    y_n = np.array(Y_N)
    corners = []

    windows = []
    labels = []
    num_transf = x_n.size * y_n.size

    window_size = getImageSize(resolution_lvl)

    transf = 0
    for delta_x in x_n:
        for delta_y in y_n:
            window = cropAndChangeResolution(image, image_name, x+delta_x, y+delta_y, window_size, resolution_lvl)
            windows.append(np.array(window))
            corners.append(np.array([x+delta_x, y+delta_y]))
            label = np.zeros(num_transf, 'uint8')
            label[transf] = 1
            labels.append(label)
            transf += 1
    return  np.stack(windows), np.stack(labels), np.uint16(np.stack(corners))


def get_callib_samples(path, radius, net):
    resolution_lvl = get_resolution_level(net)
    file_names = os.listdir(path + "Train/")
    positive_samples = []
    corners = []
    labels = []

    for image_name in file_names:
        # Ignore OSX files
        if image_name != ".DS_Store" and image_name != "train.csv":
            print ("Processing ", image_name)
            image = Image.open(path + "Train/" + image_name)
            #CHECK IF THERE IS A SEA LION IN THE IMAGE
            coordinates = extractCoordinates(path, image_name)
            classes = enumerate(CLASSES)
            for class_index, class_name in classes:
                for lion in range(len(coordinates[class_name][image_name])):
                    # Only consider sea lions within radius pixels from the edge
                    # TODO consider edge cases
                    x = coordinates[class_name][image_name][lion][0]
                    y = coordinates[class_name][image_name][lion][1]
                    if x > radius and x < (image.size[0] - radius) and y > radius and y < (image.size[1] - radius):
                        top_x = x - radius
                        top_y = y - radius
                        data, label, corner = get_shifted_windows(image, image_name, top_x, top_y, resolution_lvl)
                        positive_samples.append(data)
                        labels.append(label)
                        corners.append(corner)
    # Concatenate
    positive_samples = np.float64(np.concatenate(positive_samples))
    labels = np.uint8(np.concatenate(labels))
    corners = np.uint16(np.concatenate(corners))
    return positive_samples, labels, corners



def create_callib_dataset(path, window_size, net):
    import h5py
    radius = round(window_size / 2)
    # Get positive samples
    X, y, corners = get_callib_samples(path, radius, net)
    print (X.shape )
    print (corners.shape)
    print (y.shape)
    # Shuffle data
    X, corners, y = unison_shuffled_copies(X, corners, y)
    #Normalize
    X /= 255.0
    # Save to disk
    f = h5py.File('Datasets/data_callib'+str(net)+'_small.h5', 'w')
    # Create dataset to store images
    X_dset = f.create_dataset('data', X.shape, dtype='f')
    X_dset[:] = X
    # Create dataset to store corners
    corners_dset = f.create_dataset('corners', corners.shape, dtype='i')
    corners_dset[:] = corners
    # Create dataset to store labels
    y_dset = f.create_dataset('labels', y.shape, dtype='i')
    y_dset[:] = y
    f.close()



"""Testing"""
if __name__ == '__main__':

    #print "POS", get_positive_samples(PATH, ORIGINAL_WINDOW_DIM / 2, 1)
    #print "NEG", get_negative_samples(PATH, ORIGINAL_WINDOW_DIM / 2, 1)

    # print("COMBINED\n", create_net_dataset(PATH, ORIGINAL_WINDOW_DIM / 2, 1))
    
    # # Instantiate HDF5Matrix for the training set
    #X_train = HDF5Matrix('data_net1_small.h5', 'data', start=0, end=100)
    #y_train = HDF5Matrix('data_net1_small.h5', 'labels', start=0, end=100)
    #print X_train.shape
    #print y_train.shape

    create_callib_dataset(PATH, ORIGINAL_WINDOW_DIM, 1)
