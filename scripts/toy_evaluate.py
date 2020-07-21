from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

import settings
from toy_dataset_loader import TunnelDataset, ToTensor


def generate_subset_evaluation_plots(data_subset_type, model, num_samples_to_eval):
    dataset = TunnelDataset(root_dir=settings.TOY_IMAGE_DIR,
                            data_subset_type=data_subset_type,
                            transform=transforms.Compose([ToTensor()]))
    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=False, num_workers=1)
    subset_fig_path = settings.RESULTS_DIR + data_subset_type + "/"
    Path(subset_fig_path).mkdir(parents=True, exist_ok=True)

    for i in range(num_samples_to_eval):
        img = data_loader.dataset[i]['image'].unsqueeze_(0).unsqueeze_(0)
        speed_labels = data_loader.dataset[i]['speeds']

        speed_estimate = model(img).detach().numpy()
        plt.figure(figsize=(15, 5))
        plt.plot(speed_labels, label="Ground truth")
        plt.plot(speed_estimate[0], label="Prediction")
        plt.xlabel("Index")
        plt.ylabel("Speed (or width)")
        plt.title("%s%s%s" % ("Performance on ", data_subset_type, " set example"))
        plt.legend()
        plt.grid()
        plt.savefig("%s%s%s%i%s" % (subset_fig_path, data_subset_type, "-performance_", i, ".png"))
        plt.close()


def calculate_rmse(data_subset_type, model):
    dataset = TunnelDataset(root_dir=settings.TOY_IMAGE_DIR,
                            data_subset_type=data_subset_type,
                            transform=transforms.Compose([ToTensor()]))
    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=False, num_workers=1)
    print("RMSE for", data_subset_type, "set:")
    cumulative_rmse = 0

    for i in range(len(data_loader)):
        img = data_loader.dataset[i]['image'].unsqueeze_(0).unsqueeze_(0)
        speed_labels = data_loader.dataset[i]['speeds'].numpy()
        speed_estimate = model(img).detach().numpy()
        rmse = np.sqrt(np.mean(np.square(speed_labels - speed_estimate) / len(speed_labels)))
        cumulative_rmse += rmse
    print(cumulative_rmse / len(data_loader))


def do_quick_evaluation():
    start_time = time.time()
    model = settings.MODEL
    model.load_state_dict(torch.load(settings.MODEL_PATH))
    model.eval()
    print("Loaded model from", settings.MODEL_PATH, "-> ready to evaluate.")

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
    do_quick_evaluation()


if __name__ == "__main__":
    main()
