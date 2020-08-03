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
TOTAL_SAMPLES = 4000
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 1 - (TRAIN_RATIO + VAL_RATIO)

# Radar scan dataset parameters
RADAR_SCAN_DIMENSION = 512
RADAR_IMAGE_DIMENSION = 64
RADAR_RESOLUTION_SCALING_FACTOR = int(RADAR_SCAN_DIMENSION / RADAR_IMAGE_DIMENSION)
DO_MAX_POOLING = True
POSE_WINDOW_SIZE = 10
ODOMETRY_SPEED_MEAN = 1.31
ODOMETRY_SPEED_STD_DEV = 0.84

# Models
LENET_STR = "LeNet"
RESNET_STR = "ResNet"

ARCHITECTURE_TYPE = RESNET_STR
MODEL = None

if ARCHITECTURE_TYPE == LENET_STR:
    MODEL = LeNet()
elif ARCHITECTURE_TYPE == RESNET_STR:
    MODEL = resnet18()

# Paths
ROOT_DIR = "/workspace/data/speed-from-image/"
RADAR_IMAGE_DIR = ROOT_DIR + "radar-images/"
MODEL_DIR = ROOT_DIR + "models/"
RESULTS_DIR = ROOT_DIR + "evaluation/" + ARCHITECTURE_TYPE + "/"
RADAR_DATASET_PATH = "/workspace/data/RadarDataLogs/2017-08-18-11-21-04-oxford-10k-with-radar-1/logs/radar/cts350x/" \
                     "2017-08-18-10-21-06"
RAW_SCAN_MONOLITHIC = RADAR_DATASET_PATH + "/cts350x_raw_scan.monolithic"
RADAR_CONFIG = RADAR_DATASET_PATH + "/cts350x_config.monolithic"
