# Project-specific settings live here.
# from lenet import LeNet
# from resnet import resnet18

import torch
import numpy as np
import random

# Ensure reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# data subset names
TRAIN_SUBSET = "training"
VAL_SUBSET = "validation"
TEST_SUBSET = "test"

# General dataset parameters
TOTAL_SAMPLES = 1000
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 1 - (TRAIN_RATIO + VAL_RATIO)

# Maze parameters
MAZE_IMAGE_DIMENSION = 64
MAZE_ROBOT_RADIUS = int(MAZE_IMAGE_DIMENSION/32)
MAX_ITERATIONS = 200
NUM_OBSTACLES = 50
MAP_SIZE = 32
OBSTACLE_INFLUENCE_RADIUS = 5
GOAL_WEIGHT = 1
OBSTACLE_WEIGHT = 1
VELOCITY_LIMIT = 1
# MAZE_SPEED_MEAN = 11.44
# MAZE_SPEED_STD_DEV = 2.28

# Models
# LENET_STR = "LeNet"
# RESNET_STR = "ResNet"
#
# ARCHITECTURE_TYPE = LENET_STR
# MODEL = None
#
# if ARCHITECTURE_TYPE == LENET_STR:
#     MODEL = LeNet()
# elif ARCHITECTURE_TYPE == RESNET_STR:
#     MODEL = resnet18()

# Paths
ROOT_DIR = "/workspace/data/speed-from-image/"
MAZE_IMAGE_DIR = ROOT_DIR + "maze-images/"
# MAZE_MODEL_DIR = ROOT_DIR + "maze-models/"
# MAZE_RESULTS_DIR = ROOT_DIR + "maze-evaluation/" + ARCHITECTURE_TYPE + "/"