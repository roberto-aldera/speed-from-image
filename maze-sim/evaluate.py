from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

import settings
from dataset_loader import MazeDataset, ToTensor

# data_transform = transforms.Compose([ToTensor(), Normalise()])
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
        img = data_loader.dataset[i]['image'].unsqueeze_(0).unsqueeze_(0)
        dx_labels = data_loader.dataset[i]['dx_data']

        dx_estimate = ((model(img).detach().numpy()) * settings.MAZE_SPEED_STD_DEV) + settings.MAZE_SPEED_MEAN
        plt.figure(figsize=(15, 5))
        plt.plot(dx_labels, label="Ground truth")
        plt.plot(dx_estimate[0], label="Prediction")
        plt.ylim(0, 1)
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
        img = data_loader.dataset[i]['image'].unsqueeze_(0).unsqueeze_(0)
        dx_labels = data_loader.dataset[i]['dx_data'].numpy()
        dx_estimate = ((model(img).detach().numpy()) * settings.MAZE_SPEED_STD_DEV) + settings.MAZE_SPEED_MEAN
        rmse = np.sqrt(np.mean(np.square(dx_labels - dx_estimate) / len(dx_labels)))
        cumulative_rmse += rmse
    print(cumulative_rmse / len(data_loader))


def do_quick_evaluation(model_path):
    start_time = time.time()
    model = settings.MODEL
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Loaded model from", model_path, "-> ready to evaluate.")

    print("Generating evaluation plots...")
    num_samples = 5
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
    path_to_model = "%s%s%s" % (settings.MAZE_MODEL_DIR, settings.ARCHITECTURE_TYPE, "_checkpoint.pt")
    do_quick_evaluation(model_path=path_to_model)


if __name__ == "__main__":
    main()
