N_CALIBRATION_TRANSFORMATIONS = 9
TRAIN_SPLIT = 0.8
ORIGINAL_WINDOW_DIM = 32 #MUST BE dividable BY 4!!
PADDING_SLIDING_WINDOW = 5
CALIBRATION_THRESHOLD = 0.4

# Calibration offset vectors
X_N = [-5, 0, 5]
Y_N = [-5, 0, 5]

CLASSES = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]

REGULARIZATION_CALIBRATION_1 = 0.001
REGULARIZATION_CALIBRATION_2 = 0.001
REGULARIZATION_CALIBRATION_3 = 0.08

NUM_NEG_SAMPLES = 30 # Number of negative samples per image (should be >100)

PATH = ""

OVERLAPPING_THRES = 0.3

