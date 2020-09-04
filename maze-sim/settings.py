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
TOTAL_SAMPLES = 2000
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 1 - (TRAIN_RATIO + VAL_RATIO)

# Maze parameters
MAZE_IMAGE_DIMENSION = 32
MAP_SIZE = MAZE_IMAGE_DIMENSION
MAX_ITERATIONS = 20
MAX_NUM_OBSTACLES = 100
MIN_DISTANCE_BETWEEN_OBSTACLES = 1
EXCLUSION_ZONE_RADIUS = 2
OBSTACLE_INFLUENCE_RADIUS = 20
GOAL_FORCE_MULTIPLIER = 0.5
OBSTACLE_FORCE_MULTIPLIER = 0.5
VELOCITY_LIMIT = 0.5
NUM_POSE_DIMS = 3
ADDITIONAL_OBSTACLE_VISUAL_WEIGHT = 0
ADDITIONAL_ROBOT_VISUAL_WEIGHT = 1
MAZE_SPEED_MEAN = np.array([4.55932844e-01, 1.62530714e-04, - 4.14275714e-04])
MAZE_SPEED_STD_DEV = np.array([0.07888052, 0.04509365, 0.16137534])
# MAZE_SPEED_MEAN = np.array([0, 0, 0])
# MAZE_SPEED_STD_DEV = np.array([1, 1, 1])

# Models
# ARCHITECTURE_TYPE = "lenet"
ARCHITECTURE_TYPE = "resnet"

# Training parameters
MAX_EPOCHS = 5
LEARNING_RATE = 1e-4
BATCH_SIZE = 64

# Paths
ROOT_DIR = "/workspace/data/speed-from-image/"
MAZE_IMAGE_DIR = ROOT_DIR + "maze-images/"
MAZE_MODEL_DIR = ROOT_DIR + "maze-models/"
MAZE_RESULTS_DIR = ROOT_DIR + "maze-evaluation/"
