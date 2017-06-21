import sys, pdb
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import ModelCheckpoint, Callback
from PIL import Image
import matplotlib.pyplot as plt
from calibration_nets import *
from global_variables import *
from make_datasets import *
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

def train_calibr_net1():
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.metrics = []

        def on_epoch_end(self, epoch, logs={}):
            self.metrics.append(logs)
            np.save(PATH + 'Results/loss_calibration_net1.npy', np.array(self.metrics))

    """Train the first binary net and save training data means and best model weights.
    """
    # Load data
    X_data = HDF5Matrix(PATH + 'Datasets/data_calib1_small.h5', 'data')
    y_data = HDF5Matrix(PATH + 'Datasets/data_calib1_small.h5', 'labels')
    # Split into training and validation sets
    X_train, y_train, X_test, y_test = split_data(X_data, y_data, TRAIN_SPLIT)
    # Zero center
    means = np.mean(X_train, axis = 0)
    X_train -= means
    X_test -= means
    # Save means (for testing)
    np.save(PATH + 'Datasets/means_calib1.npy',means)
    history = LossHistory()
    # Checkpoint (for saving the weights)
    filepath = PATH + 'Weights/weights_calibration_net1.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, 
            save_weights_only=True, mode='max')
    callbacks_list = [checkpoint, history]
    # Train model (and save the weights)

    # prepare data augmentation configuration
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=50,
        width_shift_range=0.,
        height_shift_range=0.,
        shear_range=0.,
        zoom_range=0.05,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=None,
        preprocessing_function=None)

    bestlr = 0
    bestAcc = 0
    bestRegularizationTerm = 0
    for i in range(150):
        learningRate = np.random.uniform(0.01, 0.00001, 1)
        print(str(i + 1) + ' - Learning Rate: ', learningRate)
        regularization_term = np.random.uniform(0.1, 0.0001, 1)
        print(str(i + 1) + ' - Regularization Term: ', regularization_term)
        # Create model
        model = calibration_net_3(Input(shape=(X_train.shape[1], X_train.shape[2], 3)), N_CALIBRATION_TRANSFORMATIONS,
                                  regularization_term)
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        print(model.summary())
        opt = Adam(lr=learningRate, decay=1e-5)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        best_accuracy_model = 0
        for e in range(5):
            print('Epoch', e + 1)
            batches = 0
            for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=32):
                model.fit(x_batch, y_batch,
                          validation_data=(X_test, y_test),
                          shuffle='batch',  # Have to use shuffle='batch' or False with HDF5Matrix
                          verbose=0, )
                loss, acc = model.evaluate(X_test, y_test, verbose=0)
                if acc > best_accuracy_model:
                    best_accuracy_model = acc
                batches += 1
                if batches >= len(X_train) / 32:
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    break
        if best_accuracy_model > bestAcc:
            bestlr = learningRate
            bestRegularizationTerm = regularization_term
            bestAcc = best_accuracy_model
    print('best learning rate is ' + str(bestlr))
    print('best regularization term is ' + str(bestRegularizationTerm))
    print('best accuracy is: ' + str(bestAcc))

    # Now, further train with the best value
    opt = Adam(lr=bestlr, decay=1e-5)
    # Create model
    model = calibration_net_1(Input(shape=(X_train.shape[1], X_train.shape[2], 3)), N_CALIBRATION_TRANSFORMATIONS,
                              bestRegularizationTerm)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    for e in range(100):
        print('Epoch', e + 1)
        batches = 0
        for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=32):
            model.fit(x_batch, y_batch,
                      validation_data=(X_test, y_test),
                      shuffle='batch',  # Have to use shuffle='batch' or False with HDF5Matrix
                      verbose=1,
                      epochs=1,
                      callbacks=callbacks_list)
            batches += 1
            if batches >= len(X_train) / 32:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break


def train_calibr_net2():
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.metrics = []

        def on_epoch_end(self, epoch, logs={}):
            self.metrics.append(logs)
            np.save(PATH + 'Results/loss_calibration_net2.npy', np.array(self.metrics))

    """Train the second binary net and save training data means and best model weights.
    """
    # Load data
    X_data = HDF5Matrix(PATH + 'Datasets/data_calib2_small.h5', 'data')
    y_data = HDF5Matrix(PATH + 'Datasets/data_calib2_small.h5', 'labels')
    # Split into training and validation sets
    X_train, y_train, X_test, y_test = split_data(X_data, y_data, TRAIN_SPLIT)
    # Zero center
    means = np.mean(X_train, axis = 0)
    X_train -= means
    X_test -= means
    # Save means (for testing)
    np.save(PATH + 'Datasets/means_calib2.npy',means)
    history = LossHistory()
    # Checkpoint (for saving the weights)
    filepath = PATH + 'Weights/weights_calibration_net2.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, 
            save_weights_only=True, mode='max')
    callbacks_list = [checkpoint, history]
    # Train model (and save the weights)
    # prepare data augmentation configuration
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=50,
        width_shift_range=0.,
        height_shift_range=0.,
        shear_range=0.,
        zoom_range=0.05,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=None,
        preprocessing_function=None)

    bestlr = 0
    bestAcc = 0
    bestRegularizationTerm = 0
    for i in range(150):
        learningRate = np.random.uniform(0.01, 0.00001, 1)
        print(str(i + 1) + ' - Learning Rate: ', learningRate)
        regularization_term = np.random.uniform(0.1, 0.0001, 1)
        print(str(i + 1) + ' - Regularization Term: ', regularization_term)
        # Create model
        model = calibration_net_3(Input(shape=(X_train.shape[1], X_train.shape[2], 3)), N_CALIBRATION_TRANSFORMATIONS,
                                  regularization_term)
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        print(model.summary())
        opt = Adam(lr=learningRate, decay=1e-5)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        best_accuracy_model = 0
        for e in range(5):
            print('Epoch', e + 1)
            batches = 0
            for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=32):
                model.fit(x_batch, y_batch,
                          validation_data=(X_test, y_test),
                          shuffle='batch',  # Have to use shuffle='batch' or False with HDF5Matrix
                          verbose=0, )
                loss, acc = model.evaluate(X_test, y_test, verbose=0)
                if acc > best_accuracy_model:
                    best_accuracy_model = acc
                batches += 1
                if batches >= len(X_train) / 32:
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    break
        if best_accuracy_model > bestAcc:
            bestlr = learningRate
            bestRegularizationTerm = regularization_term
            bestAcc = best_accuracy_model
    print('best learning rate is ' + str(bestlr))
    print('best regularization term is ' + str(bestRegularizationTerm))
    print('best accuracy is: ' + str(bestAcc))

    # Now, further train with the best value
    opt = Adam(lr=bestlr, decay=1e-5)
    # Create model
    model = calibration_net_2(Input(shape=(X_train.shape[1], X_train.shape[2], 3)), N_CALIBRATION_TRANSFORMATIONS,
                              bestRegularizationTerm)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    for e in range(100):
        print('Epoch', e + 1)
        batches = 0
        for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=32):
            model.fit(x_batch, y_batch,
                      validation_data=(X_test, y_test),
                      shuffle='batch',  # Have to use shuffle='batch' or False with HDF5Matrix
                      verbose=1,
                      epochs=1,
                      callbacks=callbacks_list)
            batches += 1
            if batches >= len(X_train) / 32:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break


def train_calibr_net3():
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.metrics = []

        def on_epoch_end(self, epoch, logs={}):
            self.metrics.append(logs)
            np.save(PATH + 'Results/loss_calibration_net3.npy', np.array(self.metrics))

    """Train the third binary net and save training data means and best model weights.
    """
    # Load data
    X_data = HDF5Matrix(PATH + 'Datasets/data_calib3_small.h5', 'data')
    y_data = HDF5Matrix(PATH + 'Datasets/data_calib3_small.h5', 'labels')
    # Split into training and validation sets
    X_train, y_train, X_test, y_test = split_data(X_data, y_data, TRAIN_SPLIT)
    # Zero center
    means = np.mean(X_train, axis = 0)
    X_train -= means
    X_test -= means
    # Save means (for testing)
    np.save(PATH + 'Datasets/means_calib3.npy',means)
    # Create model
    history = LossHistory()
    # Checkpoint (for saving the weights)
    filepath = PATH + 'Weights/weights_calibration_net3.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, 
            save_weights_only=True, mode='max')
    callbacks_list = [checkpoint, history]
    # prepare data augmentation configuration
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=50,
        width_shift_range=0.,
        height_shift_range=0.,
        shear_range=0.,
        zoom_range=0.05,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=None,
        preprocessing_function=None)

    bestlr = 0
    bestAcc = 0
    bestRegularizationTerm = 0
    for i in range(100):
        learningRate = np.random.uniform(0.01, 0.00001, 1)
        print(str(i + 1) + ' - Learning Rate: ', learningRate)
        regularization_term = np.random.uniform(0.1, 0.0001, 1)
        print(str(i + 1) + ' - Regularization Term: ', regularization_term)
        # Create model
        model = calibration_net_3(Input(shape=(X_train.shape[1], X_train.shape[2], 3)), N_CALIBRATION_TRANSFORMATIONS,
                                  regularization_term)
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        print(model.summary())
        opt = Adam(lr=learningRate, decay=1e-5)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        best_accuracy_model = 0
        for e in range(5):
            print('Epoch', e + 1)
            batches = 0
            for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=32):
                model.fit(x_batch, y_batch,
                          validation_data=(X_test, y_test),
                          shuffle='batch',  # Have to use shuffle='batch' or False with HDF5Matrix
                          verbose=0, )
                loss, acc = model.evaluate(X_test, y_test, verbose=0)
                if acc > best_accuracy_model:
                    best_accuracy_model = acc
                batches += 1
                if batches >= len(X_train) / 32:
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    break
        if best_accuracy_model > bestAcc:
            bestlr = learningRate
            bestRegularizationTerm = regularization_term
            bestAcc = best_accuracy_model
    print('best learning rate is ' + str(bestlr))
    print('best regularization term is ' + str(bestRegularizationTerm))
    print('best accuracy is: ' + str(bestAcc))

    # Now, further train with the best value
    opt = Adam(lr=bestlr, decay=1e-5)
    # Create model
    model = calibration_net_3(Input(shape=(X_train.shape[1], X_train.shape[2], 3)), N_CALIBRATION_TRANSFORMATIONS,
                              bestRegularizationTerm)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    for e in range(100):
        print('Epoch', e + 1)
        batches = 0
        for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=32):
            model.fit(x_batch, y_batch,
                      validation_data=(X_test, y_test),
                      shuffle='batch',  # Have to use shuffle='batch' or False with HDF5Matrix
                      verbose=1,
                      epochs=1,
                      callbacks=callbacks_list)
            batches += 1
            if batches >= len(X_train) / 32:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

"""Testing"""
if __name__ == '__main__':
    try:
        arg1 = sys.argv[1]
    except IndexError:
        print("Command line argument missing. Usage: train_calibr_nets.py <net number>")
        sys.exit(1)
    if arg1 == '1':
        train_calibr_net1()
    elif arg1 == '2':
        train_calibr_net2()
    elif arg1 == '3':
        train_calibr_net3()
    else:
        print("Wrong command line argument. Must be a value between 1-3.")



