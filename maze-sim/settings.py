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
TOTAL_SAMPLES = 10000
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TRAIN_SET_SIZE = int(TOTAL_SAMPLES * TRAIN_RATIO)
VAL_SET_SIZE = int(TOTAL_SAMPLES * VAL_RATIO)
TEST_SET_SIZE = TOTAL_SAMPLES - (TRAIN_SET_SIZE + VAL_SET_SIZE)

# Maze parameters
MAZE_IMAGE_DIMENSION = 32
MAP_SIZE = MAZE_IMAGE_DIMENSION
MAX_ITERATIONS = 20
MAX_NUM_OBSTACLES = 5
MIN_DISTANCE_BETWEEN_OBSTACLES = 1
EXCLUSION_ZONE_RADIUS = 1
MAX_OBSTACLE_X_POSITION_FROM_CENTRE = 10
MIN_OBSTACLE_Y_POSITION = 20
OBSTACLE_INFLUENCE_RADIUS = MAZE_IMAGE_DIMENSION
GOAL_FORCE_MULTIPLIER = 0.5
OBSTACLE_FORCE_MULTIPLIER = 5
VELOCITY_LIMIT = 0.5
NUM_POSE_DIMS = 3
ADDITIONAL_OBSTACLE_VISUAL_WEIGHT = 1
ADDITIONAL_ROBOT_VISUAL_WEIGHT = 1
# MAZE_SPEED_MEAN = np.array([4.55932844e-01, 1.62530714e-04, - 4.14275714e-04])
# MAZE_SPEED_STD_DEV = np.array([0.07888052, 0.04509365, 0.16137534])
# MAZE_SPEED_MEAN = np.array([0, 0, 0])
# MAZE_SPEED_STD_DEV = np.array([1, 1, 1])
MAZE_SPEED_MEAN = np.array([3.37913594e-01, -3.45118750e-04, 1.04003125e-04])
MAZE_SPEED_STD_DEV = np.array([0.12659675, 0.02111636, 0.08515108])

# Models
# ARCHITECTURE_TYPE = "lenet"
ARCHITECTURE_TYPE = "resnet"

# Training parameters
NUM_CPUS = 8
MAX_EPOCHS = 5
LEARNING_RATE = 1e-4
BATCH_SIZE = 64

# Paths
ROOT_DIR = "/workspace/data/speed-from-image/"
MAZE_IMAGE_DIR = ROOT_DIR + "maze-images/"
MAZE_MODEL_DIR = ROOT_DIR + "maze-models/"
MAZE_RESULTS_DIR = ROOT_DIR + "maze-evaluation/"
