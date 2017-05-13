from keras.utils.io_utils import HDF5Matrix
from PIL import Image
import numpy as np
import os
import time


from usefulFunctions import *


PATH = "Data/"
WINDOW_SIZE = 100
NUM_NEG_SAMPLES = 130 # Number of negative samples per image

def get_positive_samples(path, radius, net):
    """Get an array of positive samples (windows containing lions), their upper left corner 
    coordinates and their labels (both in binary and multiclass one-hot representation)
    """
    resolution_lvl = get_resolution_level(net)
    file_names = os.listdir(path + "Train/")
    positive_samples = []
    corners = []
    binary_labels = []
    multiclass_labels = []

    for image_name in file_names[1:3]:
        # Ignore OSX files
        if image_name != ".DS_Store":
            print "Processing ", image_name
            image = Image.open(path + "Train/" + image_name)
            #CHECK IF THERE IS A SEA LION IN THE IMAGE
            coordinates = extractCoordinates(path, image_name)
            classes = enumerate(["adult_males", "subadult_males", "adult_females", "juveniles", "pups"])
            for class_index, class_name in classes:
                for lion in range(len(coordinates[class_name][image_name])):
                    # Only consider sea lions within radius pixels from the edge
                    # TODO consider edge cases
                    x = coordinates[class_name][image_name][lion][0]
                    y = coordinates[class_name][image_name][lion][1]
                    if x > radius and x < (image.size[0] - radius) and y > radius and y < (image.size[1] - radius):
                        # Crop window of chosen resolution level
                        window = cropAndChangeResolution(path, image_name, x - radius, y - radius, radius * 2, resolution_lvl)
                        # Append
                        positive_samples.append(np.array(window))
                        corners.append(np.array([x - radius, y - radius]))
                        binary_labels.append([1,0])
                        multiclass =  np.zeros(5, 'uint8')
                        multiclass[class_index] = 1
                        multiclass_labels.append(multiclass)
    # Concatenate
    positive_samples = np.float64(np.stack(positive_samples))
    corners = np.uint16(np.stack(corners))
    binary_labels = np.uint8(np.array(binary_labels))
    multiclass_labels = np.uint8(np.stack(multiclass_labels))
    return positive_samples, corners, binary_labels, multiclass_labels


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
        if image_name != ".DS_Store":
            print "Processing ", image_name
            image = Image.open(path + "Train/" + image_name)
            coordinates = extractCoordinates(path, image_name)
            for it in range(15):
                # Upper left corner coordinates
                x = np.random.uniform(0, image.size[0] - 2 * radius)
                y = np.random.uniform(0, image.size[1] - 2 * radius)
                window = cropAndChangeResolution(path, image_name, x, y, radius * 2, resolution_lvl)
                label = getLabel(image_name, coordinates, x, y, radius * 2)
                # Append negative samples 
                if label == [0, 1]:
                    negative_samples.append(np.array(window))
                    corners.append(np.array([x, y]))
                    labels.append(label)
    # Concatenate
    negative_samples = np.float64(np.stack(negative_samples))
    corners = np.uint16(np.stack(corners))
    labels = np.uint8(np.array(labels))

    return negative_samples, corners, labels


def unison_shuffled_copies(a, b, c, d=None):
    assert len(a) == len(b) and len(b) == len(c)
    p = np.random.permutation(len(a))
    if d != None:
        assert len(a) == len(d)
        return a[p], b[p], c[p], d[p]
    else:
        return a[p], b[p], c[p]



def create_net_dataset(path, window_size, net):
    import h5py
    radius = round(window_size / 2)
    # Get positive samples
    pos_samples, pos_corners, pos_labels, _ = get_positive_samples(path, radius, net)
    print pos_samples.shape 
    print pos_corners.shape
    print pos_labels.shape
    # Get negative samples
    neg_samples, neg_corners, neg_labels = get_negative_samples(path, radius, net)
    print neg_samples.shape 
    print neg_corners.shape
    print neg_labels.shape
    # Concatenate positive and negative
    X = np.concatenate((pos_samples, neg_samples))
    corners = np.concatenate((pos_corners, neg_corners))
    y = np.concatenate((pos_labels, neg_labels))
    # Shuffle data
    X, corners, y = unison_shuffled_copies(X, corners, y)
    # Save to disk
    f = h5py.File('data_net'+str(net)+'_small.h5', 'w')
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


def get_shifted_windows(path, image_name, x, y, resolution_lvl):

    # Offset vectors
    x_n = np.array([-6, -4, -2, 0, 2, 4, 6])
    y_n = np.array([-6, -4, -2, 0, 2, 4, 6])

    windows = []
    labels = []
    num_transf = x_n.size * y_n.size

    window_size = getImageSize(resolution_lvl)

    transf = 0
    for delta_x in x_n:
        for delta_y in y_n:
            window = cropAndChangeResolution(path, image_name, x+delta_x, y+delta_y, window_size, resolution_lvl)
            windows.append(np.array(window))
            label = np.zeros(num_transf, 'uint8')
            label[transf] = 1
            labels.append(label)
            transf += 1
    return  np.stack(windows), np.stack(labels)


def get_callib_samples(path, radius, net):
    resolution_lvl = get_resolution_level(net)
    file_names = os.listdir(path + "Train/")
    positive_samples = []
    corners = []
    labels = []

    for image_name in file_names[1:3]:
        # Ignore OSX files
        if image_name != ".DS_Store":
            print "Processing ", image_name
            image = Image.open(path + "Train/" + image_name)
            #CHECK IF THERE IS A SEA LION IN THE IMAGE
            coordinates = extractCoordinates(path, image_name)
            classes = enumerate(["adult_males", "subadult_males", "adult_females", "juveniles", "pups"])
            for class_index, class_name in classes:
                for lion in range(len(coordinates[class_name][image_name])):
                    # Only consider sea lions within radius pixels from the edge
                    # TODO consider edge cases
                    x = coordinates[class_name][image_name][lion][0]
                    y = coordinates[class_name][image_name][lion][1]
                    if x > radius and x < (image.size[0] - radius) and y > radius and y < (image.size[1] - radius):
                        top_x = x - radius
                        top_y = y - radius
                        data, label = get_shifted_windows(path, image_name, top_x, top_y, resolution_lvl)
                        print "\t ", len(data), "shifted windows"
                        positive_samples.append(data)
                        labels.append(label)
    # Concatenate
    positive_samples = np.stack(positive_samples)
    labels = np.stack(labels)
    return positive_samples, labels



def create_callib_dataset(path, window_size, net):
    import h5py
    radius = round(window_size / 2)
    # Get positive samples
    X, corners, y = get_callib_samples(path, radius, net)
    print X.shape 
    print corners.shape
    print y.shape
    # Shuffle data
    X, corners, y = unison_shuffled_copies(X, corners, y)
    # Save to disk
    f = h5py.File('data_callib'+str(net)+'_small.h5', 'w')
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

    # pos_samples, pos_corners, pos_labels, _ = get_positive_samples(PATH, WINDOW_SIZE / 2, 1)
    # print pos_samples.shape 
    # print pos_corners.shape
    # print pos_labels.shape

    # neg_samples, neg_corners, neg_labels = get_negative_samples(PATH, WINDOW_SIZE / 2, 1)
    # print neg_samples.shape 
    # print neg_corners.shape
    # print neg_labels.shape

    create_net_dataset(PATH, WINDOW_SIZE / 2, 1)
    # # Instantiate HDF5Matrix for the training set
    X_train = HDF5Matrix('data_net1_small.h5', 'data', start=0, end=100)
    y_train = HDF5Matrix('data_net1_small.h5', 'labels', start=0, end=100)
    print X_train.shape
    print y_train.shape

    # Zero center

    # Likewise for the test set
    # X_test = HDF5Matrix('data_net1_small.h5', 'data', start=1000, end=1200)
    # y_test = HDF5Matrix('data_net1_small.h5', 'labels', start=1000, end=1200)

    # print "checkpoint2"
    # positive_samples, labels = get_callib_data(PATH, WINDOW_SIZE / 2, 1)


# def create_training_dataset():
#     import h5py
#     X = np.random.randn(200,10).astype('float32')

#     y = np.random.randint(0, 2, size=(200,1))
#     f = h5py.File('train_small.h5', 'w')
#     # Creating dataset to store features
#     X_dset = f.create_dataset('my_data', (200,10), dtype='f')
#     X_dset[:] = X
#     # Creating dataset to store labels
#     y_dset = f.create_dataset('my_labels', (200,1), dtype='i')
#     y_dset[:] = y
#     f.close()

# create_dataset()

# # Instantiating HDF5Matrix for the training set, which is a slice of the first 150 elements
# X_train = HDF5Matrix('test.h5', 'my_data', start=0, end=150)
# y_train = HDF5Matrix('test.h5', 'my_labels', start=0, end=150)

# # Likewise for the test set
# X_test = HDF5Matrix('test.h5', 'my_data', start=150, end=200)
# y_test = HDF5Matrix('test.h5', 'my_labels', start=150, end=200)

# # HDF5Matrix behave more or less like Numpy matrices with regards to indexing
# print(y_train[10])
# # But they do not support negative indices, so don't try print(X_train[-1])

# model = Sequential()
# model.add(Dense(64, input_shape=(10,), activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='sgd')

# # Note: you have to use shuffle='batch' or False with HDF5Matrix
# model.fit(X_train, y_train, batch_size=32, shuffle='batch')

# model.evaluate(X_test, y_test, batch_size=32)

