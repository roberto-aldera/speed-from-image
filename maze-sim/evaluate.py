from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

import settings
from dataset_loader import MazeDataset, ToTensor

data_transform_for_evaluation = transforms.Compose([ToTensor()])


def generate_subset_evaluation_plots(data_subset_type, model, num_samples_to_eval):
    dataset = MazeDataset(root_dir=settings.MAZE_IMAGE_DIR,
                          data_subset_type=data_subset_type,
                          transform=data_transform_for_evaluation)
    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=False, num_workers=1)
    subset_fig_path = settings.MAZE_RESULTS_DIR + data_subset_type + "/"
    Path(subset_fig_path).mkdir(parents=True, exist_ok=True)

    for i in range(num_samples_to_eval):
        img = data_loader.dataset[i]['image'].unsqueeze_(0)
        pose_labels = data_loader.dataset[i]['pose_data']
        pose_estimate = ((model(img).detach().numpy()) * settings.MAZE_SPEED_STD_DEV) + settings.MAZE_SPEED_MEAN

        plt.figure(figsize=(15, 5))
        plt.plot(pose_labels[0], 'r--', alpha=0.5, label="dx ground truth")
        plt.plot(pose_labels[1], 'g--', alpha=0.5, label="dy ground truth")
        plt.plot(pose_labels[2], 'b--', alpha=0.5, label="dth ground truth")
        plt.plot(pose_estimate[0, 0], 'r', label="dx prediction")
        plt.plot(pose_estimate[0, 1], 'g', label="dy prediction")
        plt.plot(pose_estimate[0, 2], 'b', label="dth prediction")

        plt.ylim(-1, 1)
        plt.xlabel("Index")
        plt.ylabel("dx")
        plt.title("%s%s%s" % ("Performance on ", data_subset_type, " set example"))
        plt.legend()
        plt.grid()
        plt.savefig("%s%s%s%i%s" % (subset_fig_path, data_subset_type, "-performance_", i, ".png"))
        plt.close()


def calculate_rmse(data_subset_type, model):
    dataset = MazeDataset(root_dir=settings.MAZE_IMAGE_DIR,
                          data_subset_type=data_subset_type,
                          transform=data_transform_for_evaluation)
    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=False, num_workers=1)
    print("RMSE for", data_subset_type, "set:")
    cumulative_rmse = 0

    for i in range(len(data_loader)):
        img = data_loader.dataset[i]['image'].unsqueeze_(0)
        pose_labels = data_loader.dataset[i]['pose_data'].numpy()
        pose_estimate = ((model(img).detach().numpy()) * settings.MAZE_SPEED_STD_DEV).squeeze(
            0) + settings.MAZE_SPEED_MEAN
        # print("Pose labels:", pose_labels)
        # print("Pose estimate:", pose_estimate)
        rmse = np.sqrt(np.mean(np.square(pose_labels - pose_estimate) / len(pose_labels), axis=1))
        cumulative_rmse += rmse
    print(cumulative_rmse / len(data_loader))


def do_quick_evaluation(model_path):
    start_time = time.time()
    model = settings.MODEL
    model = model.load_from_checkpoint(model_path)
    model.eval()
    print("Loaded model from", model_path, "-> ready to evaluate.")

    print("Generating evaluation plots...")
    num_samples = 10
    generate_subset_evaluation_plots(settings.TRAIN_SUBSET, model, num_samples)
    generate_subset_evaluation_plots(settings.VAL_SUBSET, model, num_samples)
    generate_subset_evaluation_plots(settings.TEST_SUBSET, model, num_samples)

    print("Calculating average RMSE (over entire subset)")
    calculate_rmse(settings.TRAIN_SUBSET, model)
    calculate_rmse(settings.VAL_SUBSET, model)
    calculate_rmse(settings.TEST_SUBSET, model)

    print("--- Evaluation execution time: %s seconds ---" % (time.time() - start_time))


def main():
    # Define a main loop to run and show some example data if this script is run as main
    path_to_model = "%s%s%s" % (settings.MAZE_MODEL_DIR, settings.ARCHITECTURE_TYPE, ".ckpt")
    do_quick_evaluation(model_path=path_to_model)


if __name__ == "__main__":
    main()
