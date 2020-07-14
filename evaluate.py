from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

from dataset_loader import TunnelDataset, ToTensor
from network import Net

start_time = time.time()

num_samples_to_eval = 5
model_path = "/Users/roberto/code/speed-from-image/models/myModel.pt"
train_fig_path = "/Users/roberto/code/speed-from-image/evaluation/training_results/"
val_fig_path = "/Users/roberto/code/speed-from-image/evaluation/validation_results/"
Path(train_fig_path).mkdir(parents=True, exist_ok=True)
Path(val_fig_path).mkdir(parents=True, exist_ok=True)

model = Net()
model.load_state_dict(torch.load(model_path))
model.eval()
print("Loaded model from", model_path, "-> ready to evaluate.")

train_dataset = TunnelDataset(root_dir="/Users/roberto/code/speed-from-image/images/",
                              data_subset_type="training",
                              transform=transforms.Compose([ToTensor()]))
train_loader = DataLoader(train_dataset, batch_size=1,
                          shuffle=False, num_workers=1)
for i in range(num_samples_to_eval):
    img = train_loader.dataset[i]['image'].unsqueeze_(0).unsqueeze_(0)
    # print(img.shape)
    speed_labels = train_loader.dataset[i]['speeds']

    speed_estimate = model(img).detach().numpy()
    plt.figure(figsize=(15, 5))
    plt.plot(speed_labels, label="Ground truth")
    plt.plot(speed_estimate[0], label="Prediction")
    plt.xlabel("Index")
    plt.ylabel("Speed (or width)")
    plt.title("Performance on training set example")
    plt.legend()
    plt.grid()
    plt.savefig("%s%s%i%s" % (train_fig_path, "train-performance_", i, ".png"))
    plt.close()
    
# Validation set
val_dataset = TunnelDataset(root_dir="/Users/roberto/code/speed-from-image/images/",
                            data_subset_type="validation",
                            transform=transforms.Compose([ToTensor()]))
val_loader = DataLoader(val_dataset, batch_size=1,
                        shuffle=False, num_workers=1)

for i in range(num_samples_to_eval):
    img = val_loader.dataset[i]['image'].unsqueeze_(0).unsqueeze_(0)
    speed_labels = val_loader.dataset[i]['speeds']

    speed_estimate = model(img).detach().numpy()
    plt.figure(figsize=(15, 5))
    plt.plot(speed_labels, label="Ground truth")
    plt.plot(speed_estimate[0], label="Prediction")
    plt.xlabel("Index")
    plt.ylabel("Speed (or width)")
    plt.title("Performance on validation set example")
    plt.legend()
    plt.grid()
    plt.savefig("%s%s%i%s" % (val_fig_path, "val-performance_", i, ".png"))
    plt.close()

# RMSE comparisons
print("Average RMSE (over entire subset)")
# Not sure if this is the best way to do this, but otherwise the batch size will be what it was in training
# and I don't know what that causes this loop to do.
train_loader = DataLoader(train_dataset, batch_size=1,
                          shuffle=False, num_workers=1)
print("Training set:")
cumulative_RMSE = 0

for i in range(len(train_loader)):
    img = train_loader.dataset[i]['image'].unsqueeze_(0).unsqueeze_(0)
    speed_labels = train_loader.dataset[i]['speeds'].numpy()
    speed_estimate = model(img).detach().numpy()
    RMSE = np.sqrt(np.mean(np.square(speed_labels - speed_estimate) / len(speed_labels)))
    cumulative_RMSE += RMSE
print(cumulative_RMSE / len(train_loader))

print("Validation set:")
cumulative_RMSE = 0

for i in range(len(val_loader)):
    img = val_loader.dataset[i]['image'].unsqueeze_(0).unsqueeze_(0)
    speed_labels = val_loader.dataset[i]['speeds'].numpy()
    speed_estimate = model(img).detach().numpy()
    RMSE = np.sqrt(np.mean(np.square(speed_labels - speed_estimate) / len(speed_labels)))
    cumulative_RMSE += RMSE
print(cumulative_RMSE / len(val_loader))

print("--- Execution time: %s seconds ---" % (time.time() - start_time))
