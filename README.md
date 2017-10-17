## Sea Lion Population Count

<img src="Data/Train/47.jpg" alt="Sample image">

We implement the adaptation of a convolutional neural network (CNN) cascade originally designed for face detection for a different visual object detection task involving in counting and classifying sea lions. The cascade architecture built on CNNs is capable of combining increasingly more complex classifiers that operate on increasingly higher resolutions. This allows background regions of the image to be quickly rejected in the low resolution stage, while spending more computation on promising sea lion like regions during higher resolution stages. The cascade can be viewed as a sea lion-focusing mechanism which ensures that discarded regions are unlikely to contain sea lions. The cascade consists of alternating detection and calibration networks, so that the output of each calibration stage is used to adjust the detection window position for input to the subsequent stage, which ultimately leads to improved localization effectiveness.

More inforation can be found in the [report](SeaLionsReport.pdf).


The Data folder is just sample Data, the real Dataset is 95GB and can be found at https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/data

## How to run the code: 

For making the dataset

$ python make_datasets.py pos1

$ python make_datasets.py pos2

$ python make_datasets.py pos3

$ python make_datasets.py neg

$ python make_datasets.py combine

$ python make_datasets.py multi

$ python make_datasets.py cal1

$ python make_datasets.py cal2

$ python make_datasets.py cal3

For training the netwoks

$ python train_binary_nets.py 1

$ python train_binary_nets.py 2

$ python train_binary_nets.py 3

$ python train_calibr_nets.py 1

$ python train_calibr_nets.py 2

$ python train_calibr_nets.py 3

$ python train_classification_net.py

For testing

$ python test_pipeline.py testFolder/

$ python count_sea_lions.py testFolder/
