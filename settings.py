# Project-specific settings live here.

# Paths
ROOT_DIR = "/Users/roberto/code/speed-from-image/"
IMAGE_DIR = ROOT_DIR + "images/"
MODEL_PATH = ROOT_DIR + "models/myModel.pt"
RESULTS_DIR = ROOT_DIR + "evaluation/"

# data subset names
TRAIN_SUBSET = "training"
VAL_SUBSET = "validation"
TEST_SUBSET = "test"

# Dataset generation
TOTAL_SAMPLES = 1000
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 1 - (TRAIN_RATIO + VAL_RATIO)
IMAGE_DIMENSION = 128
MIN_WIDTH, MAX_WIDTH = 8, 16
MIN_LENGTH, MAX_LENGTH = 15, 20
WALL_WIDTH = 2
SPEED_NOISE_LEVEL = 0.25
