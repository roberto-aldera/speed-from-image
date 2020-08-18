# Project-specific settings live here.
from lenet import LeNet
from resnet import resnet18

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
MAZE_SPEED_MEAN = 0
MAZE_SPEED_STD_DEV = 1

# Models
LENET_STR = "LeNet"
RESNET_STR = "ResNet"

ARCHITECTURE_TYPE = LENET_STR
# ARCHITECTURE_TYPE = RESNET_STR
MODEL = None

if ARCHITECTURE_TYPE == LENET_STR:
    MODEL = LeNet()
elif ARCHITECTURE_TYPE == RESNET_STR:
    MODEL = resnet18()

LEARNING_RATE = 1e-4

# Paths
ROOT_DIR = "/workspace/data/speed-from-image/"
MAZE_IMAGE_DIR = ROOT_DIR + "maze-images/"
MAZE_MODEL_DIR = ROOT_DIR + "maze-models/"
MAZE_RESULTS_DIR = ROOT_DIR + "maze-evaluation/" + ARCHITECTURE_TYPE + "/"
