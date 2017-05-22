The Data folder is just sample Data, the real Dataset is 95GB, the real data can be found https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/data

How to run the code:

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
