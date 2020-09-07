from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
from argparse import ArgumentParser
import logging

import settings
from dataset_loader import MazeDataset, ToTensor
from lenet import LeNet
from resnet import ResNet

data_transform_for_evaluation = transforms.Compose([ToTensor()])


def generate_subset_evaluation_plots(data_subset_type, model, results_path, num_samples_to_eval):
    dataset = MazeDataset(root_dir=settings.MAZE_IMAGE_DIR,
                          data_subset_type=data_subset_type,
                          transform=data_transform_for_evaluation)
    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=False, num_workers=1)
    subset_fig_path = results_path + data_subset_type + "/"
    Path(subset_fig_path).mkdir(parents=True, exist_ok=True)
    print("Saving evaluation plots to:", subset_fig_path)

    for i in range(num_samples_to_eval):
        img = data_loader.dataset[i]['image'].unsqueeze_(0)
        pose_labels = data_loader.dataset[i]['pose_data']
        pose_from_model = model(img).detach().numpy()
        pose_estimate = np.transpose(
            (np.transpose(pose_from_model.squeeze(0)) * settings.MAZE_SPEED_STD_DEV) + settings.MAZE_SPEED_MEAN)
        plt.figure(figsize=(15, 5))
        plt.plot(pose_labels[0], 'r--', alpha=0.5, label="dx ground truth")
        plt.plot(pose_labels[1], 'g--', alpha=0.5, label="dy ground truth")
        plt.plot(pose_labels[2], 'b--', alpha=0.5, label="dth ground truth")
        plt.plot(pose_estimate[0], 'r', label="dx prediction")
        plt.plot(pose_estimate[1], 'g', label="dy prediction")
        plt.plot(pose_estimate[2], 'b', label="dth prediction")

        plt.ylim(-1, 1)
        plt.xlabel("Index")
        plt.ylabel("dx")
        plt.title("%s%s%s" % ("Performance on ", data_subset_type, " set example"))
        plt.legend()
        plt.grid()
        plt.savefig("%s%s%s%i%s" % (subset_fig_path, data_subset_type, "-performance_", i, ".png"))
        plt.close()


def calculate_rmse(data_subset_type, model, logger):
    dataset = MazeDataset(root_dir=settings.MAZE_IMAGE_DIR,
                          data_subset_type=data_subset_type,
                          transform=data_transform_for_evaluation)
    data_loader = DataLoader(dataset, batch_size=1,  # not sure if batch size here needs to be only 1
                             shuffle=False, num_workers=4)
    logger.info("RMSE for " + str(len(data_loader)) + " " + data_subset_type + " set examples:")
    cumulative_rmse = 0

    for i in range(len(data_loader)):
        img = data_loader.dataset[i]['image'].unsqueeze_(0)
        pose_labels = data_loader.dataset[i]['pose_data'].numpy()
        pose_from_model = model(img).detach().numpy()
        pose_estimate = np.transpose(
            (np.transpose(pose_from_model.squeeze(0)) * settings.MAZE_SPEED_STD_DEV) + settings.MAZE_SPEED_MEAN)
        rmse = np.sqrt(np.mean(np.square(pose_labels - pose_estimate) / len(pose_labels), axis=1))
        cumulative_rmse += rmse
    logger.info(cumulative_rmse / len(data_loader))


def do_quick_evaluation(hparams, model, model_path):
    start_time = time.time()
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    results_path = settings.MAZE_RESULTS_DIR + hparams.model_name + "/" + current_time + "/"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    model = model.load_from_checkpoint(model_path)
    model.eval()
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(message)s', filename=(results_path + "rmse_results.txt"), level=logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.info("Loaded model from " + model_path + " -> ready to evaluate.")

    print("Generating evaluation plots...")
    num_samples = 10
    generate_subset_evaluation_plots(settings.TRAIN_SUBSET, model, results_path, num_samples)
    generate_subset_evaluation_plots(settings.VAL_SUBSET, model, results_path, num_samples)
    generate_subset_evaluation_plots(settings.TEST_SUBSET, model, results_path, num_samples)

    print("Calculating average RMSE (over entire subset)")

    calculate_rmse(settings.TRAIN_SUBSET, model, logger)
    calculate_rmse(settings.VAL_SUBSET, model, logger)
    calculate_rmse(settings.TEST_SUBSET, model, logger)

    print("--- Evaluation execution time: %s seconds ---" % (time.time() - start_time))


def main():
    Path(settings.MAZE_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.MAZE_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--model_name', type=str, default=settings.ARCHITECTURE_TYPE, help='lenet or resnet')

    temp_args, _ = parser.parse_known_args()
    # let the model add what it wants
    if temp_args.model_name == 'lenet':
        parser = LeNet.add_model_specific_args(parser)
    elif temp_args.model_name == 'resnet':
        parser = ResNet.add_model_specific_args(parser)

    hparams = parser.parse_args()
    path_to_model = "%s%s%s" % (settings.MAZE_MODEL_DIR, hparams.model_name, ".ckpt")

    # pick model
    model = None
    if hparams.model_name == 'lenet':
        model = LeNet(hparams)
    elif hparams.model_name == 'resnet':
        model = ResNet(hparams)

    do_quick_evaluation(hparams, model, path_to_model)


if __name__ == "__main__":
    main()
