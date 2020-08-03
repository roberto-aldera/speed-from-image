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

# Tunnel parameters
SIM_IMAGE_DIMENSION = 64
SIM_ROBOT_RADIUS = int(SIM_IMAGE_DIMENSION/32)
MIN_WIDTH, MAX_WIDTH = 8, 16
MIN_LENGTH, MAX_LENGTH = 15, 20
WALL_WIDTH = 2
SPEED_NOISE_LEVEL = 0.25
SIM_SPEED_MEAN = 11.44
SIM_SPEED_STD_DEV = 2.28

# Models
LENET_STR = "LeNet"
RESNET_STR = "ResNet"

ARCHITECTURE_TYPE = LENET_STR
MODEL = None

if ARCHITECTURE_TYPE == LENET_STR:
    MODEL = LeNet()
elif ARCHITECTURE_TYPE == RESNET_STR:
    MODEL = resnet18()

# Paths
ROOT_DIR = "/workspace/data/speed-from-image/"
SIM_IMAGE_DIR = ROOT_DIR + "sim-images/"
SIM_MODEL_DIR = ROOT_DIR + "sim-models/"
SIM_RESULTS_DIR = ROOT_DIR + "sim-evaluation/" + ARCHITECTURE_TYPE + "/"
