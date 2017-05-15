INPUT_SIZE_NET_1 = 25
N_CALIBRATION_TRANSFORMATIONS = 9
TRAIN_SPLIT = 0.8
ORIGINAL_WINDOW_DIM = 100
PADDING_SLIDING_WINDOW = 10

# Calibration offset vectors
X_N = [-30, 0, 30]
Y_N = [-30, 0, 30]

CLASSES = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]

REGULARIZATION_CALIBRATION_1 = 0.001

NUM_NEG_SAMPLES = 130 # Number of negative samples per image
PATH = "Data/"
