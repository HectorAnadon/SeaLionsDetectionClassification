INPUT_SIZE_NET_1 = 25
N_CALIBRATION_TRANSFORMATIONS = 9
TRAIN_SPLIT = 0.8
ORIGINAL_WINDOW_DIM = 48
PADDING_SLIDING_WINDOW = 10
CALIBRATION_THRESHOLD = 0.4

# Calibration offset vectors
X_N = [-15, 0, 15]
Y_N = [-15, 0, 15]

CLASSES = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]

REGULARIZATION_CALIBRATION_1 = 0.001

NUM_NEG_SAMPLES = 150 # Number of negative samples per image (should be >100) TODO: SET THIS

PATH = ""

OVERLAPPING_THRES = 0.3

