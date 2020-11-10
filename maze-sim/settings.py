# Project-specific settings live here.
import pytorch_lightning as pl
import numpy as np

# Ensure reproducibility
pl.seed_everything(0)

# data subset names
TRAIN_SUBSET = "training"
VAL_SUBSET = "validation"
TEST_SUBSET = "test"

# General dataset parameters
# TOTAL_SAMPLES = 10000
# TRAIN_RATIO = 0.8
# VAL_RATIO = 0.1
TOTAL_SAMPLES = 40000
TRAIN_RATIO = 0.9
VAL_RATIO = 0.05
TRAIN_SET_SIZE = int(TOTAL_SAMPLES * TRAIN_RATIO)
VAL_SET_SIZE = int(TOTAL_SAMPLES * VAL_RATIO)
TEST_SET_SIZE = TOTAL_SAMPLES - (TRAIN_SET_SIZE + VAL_SET_SIZE)

# Maze parameters
MAZE_IMAGE_DIMENSION = 64
MAP_SIZE = MAZE_IMAGE_DIMENSION
MAX_ITERATIONS = 10
TOTAL_POSES = MAX_ITERATIONS - 1
MAX_NUM_OBSTACLES = 40
MIN_DISTANCE_BETWEEN_OBSTACLES = 2
EXCLUSION_ZONE_RADIUS = 5
MAX_OBSTACLE_X_POSITION_FROM_CENTRE = 10
MIN_OBSTACLE_Y_POSITION = 5
OBSTACLE_INFLUENCE_RADIUS = 20  # MAZE_IMAGE_DIMENSION
GOAL_FORCE_MULTIPLIER = 15
OBSTACLE_FORCE_MULTIPLIER = 15
VELOCITY_LIMIT = (MAZE_IMAGE_DIMENSION / 2) / MAX_ITERATIONS
NOMINAL_VELOCITY = VELOCITY_LIMIT * 2
PROXIMITY_TO_OBSTACLE_CAUTION_FACTOR = 10
NUM_POSE_DIMS = 3
ADDITIONAL_OBSTACLE_VISUAL_WEIGHT = 0
ADDITIONAL_ROBOT_VISUAL_WEIGHT = 1
# MAZE_SPEED_MEAN = np.array([4.55932844e-01, 1.62530714e-04, - 4.14275714e-04])
# MAZE_SPEED_STD_DEV = np.array([0.07888052, 0.04509365, 0.16137534])
# MAZE_SPEED_MEAN = np.array([0, 0, 0])
# MAZE_SPEED_STD_DEV = np.array([1, 1, 1])
# MAZE_SPEED_MEAN = np.array([5.13417215e-01, 4.28445625e-05, 1.45100250e-04])
# MAZE_SPEED_STD_DEV = np.array([0.16595751, 0.0202896, 0.06574377])
# MAZE_SPEED_MEAN = np.array([5.12978690e-01, 3.22280556e-06, 3.32384722e-05])
# MAZE_SPEED_STD_DEV = np.array([0.16593968, 0.02033633, 0.06580356])
# MAZE_SPEED_MEAN = np.array([1.2552569524738388, 3.8745139649790974e-05, 3.2650718106607396e-05])
# MAZE_SPEED_STD_DEV = np.array([0.7978619029273148, 0.020788025206537605, 0.01757944113044208])
MAZE_SPEED_MEAN = np.array([1.2596747168288542, 3.0793499626990464e-05, 3.132957380221915e-05])
MAZE_SPEED_STD_DEV = np.array([0.7982359008979953, 0.02086976241601138, 0.017650547288471548])

# Models
# ARCHITECTURE_TYPE = "lenet"
ARCHITECTURE_TYPE = "resnet"

# Training parameters
NUM_CPUS = 8
MAX_EPOCHS = 5
LEARNING_RATE = 1e-4
BATCH_SIZE = 64

# Paths
IS_RUNNING_ON_SERVER = True

if IS_RUNNING_ON_SERVER is True:
    ROOT_DIR = "/Volumes/scratchdata/roberto/maze/"
    MAZE_IMAGE_DIR = "/workspace/maze-images/"
    MAZE_MODEL_DIR = ROOT_DIR + "models/"
    MAZE_RESULTS_DIR = ROOT_DIR + "evaluation/"
else:
    ROOT_DIR = "/workspace/data/speed-from-image/"
    MAZE_IMAGE_DIR = ROOT_DIR + "maze-images/"
    MAZE_MODEL_DIR = ROOT_DIR + "maze-models/"
    MAZE_RESULTS_DIR = ROOT_DIR + "maze-evaluation/"
