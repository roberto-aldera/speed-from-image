# Project-specific settings live here.
from network import Net
from resnet import resnet18

# data subset names
TRAIN_SUBSET = "training"
VAL_SUBSET = "validation"
TEST_SUBSET = "test"

# General dataset parameters
TOTAL_SAMPLES = 100
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 1 - (TRAIN_RATIO + VAL_RATIO)

# Tunnel parameters
TOY_IMAGE_DIMENSION = 64
MIN_WIDTH, MAX_WIDTH = 8, 16
MIN_LENGTH, MAX_LENGTH = 15, 20
WALL_WIDTH = 2
SPEED_NOISE_LEVEL = 0.25

# Radar scan dataset parameters
RADAR_IMAGE_DIMENSION = 256
POSE_WINDOW_SIZE = 3

# Models
LENET_STR = "LeNet"
RESNET_STR = "ResNet"

ARCHITECTURE_TYPE = LENET_STR
MODEL = None

if ARCHITECTURE_TYPE == LENET_STR:
    MODEL = Net()
elif ARCHITECTURE_TYPE == RESNET_STR:
    MODEL = resnet18()

# Paths
ROOT_DIR = "/workspace/data/speed-from-image/"
TOY_IMAGE_DIR = ROOT_DIR + "toy-images/"
RADAR_IMAGE_DIR = ROOT_DIR + "radar-images/"
MODEL_PATH = ROOT_DIR + "models/" + ARCHITECTURE_TYPE + ".pt"
RESULTS_DIR = ROOT_DIR + "evaluation/" + ARCHITECTURE_TYPE + "/"

RADAR_DATASET_PATH = "/workspace/data/RadarDataLogs/2017-08-18-11-21-04-oxford-10k-with-radar-1/logs/radar/cts350x/" \
                      "2017-08-18-10-21-06"
RAW_SCAN_MONOLITHIC = RADAR_DATASET_PATH + "/cts350x_raw_scan.monolithic"
RADAR_CONFIG = RADAR_DATASET_PATH + "/cts350x_config.monolithic"