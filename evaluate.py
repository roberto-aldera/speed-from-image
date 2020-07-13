from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from dataset_loader import TunnelDataset, ToTensor
from network import Net

model_path = "/Users/roberto/code/speed-from-image/models/myModel.pt"
fig_path = "/Users/roberto/code/speed-from-image/evaluation"
Path(fig_path).mkdir(parents=True, exist_ok=True)

model = Net()
model.load_state_dict(torch.load(model_path))
model.eval()
print("Loaded model from", model_path, "-> ready to evaluate.")

train_dataset = TunnelDataset(root_dir="/Users/roberto/code/speed-from-image/images/",
                              data_subset_type="training",
                              transform=transforms.Compose([ToTensor()]))
train_loader = DataLoader(train_dataset, batch_size=1,
                          shuffle=False, num_workers=1)
img_idx = 0
img = train_loader.dataset[img_idx]['image'].unsqueeze_(0).unsqueeze_(0)
# print(img.shape)
speed_labels = train_loader.dataset[img_idx]['speeds']

speed_estimate = model(img).detach().numpy()
plt.plot(speed_labels, label="Ground truth")
plt.plot(speed_estimate[0], label="Prediction")
plt.xlabel("Index")
plt.ylabel("Speed (or width)")
plt.legend()
plt.grid()
# print(speed_estimate)
plt.savefig("/Users/roberto/code/speed-from-image/evaluation/training-performance.png")

# Validation set
val_dataset = TunnelDataset(root_dir="/Users/roberto/code/speed-from-image/images/",
                            data_subset_type="validation",
                            transform=transforms.Compose([ToTensor()]))
val_loader = DataLoader(val_dataset, batch_size=1,
                        shuffle=False, num_workers=1)

for i in range(1):
    img_idx = i
    img = val_loader.dataset[img_idx]['image'].unsqueeze_(0).unsqueeze_(0)
    speed_labels = val_loader.dataset[img_idx]['speeds']

    speed_estimate = model(img).detach().numpy()
    plt.figure()
    plt.plot(speed_labels, label="Ground truth")
    plt.plot(speed_estimate[0], label="Prediction")
    plt.xlabel("Index")
    plt.ylabel("Speed (or width)")
    plt.legend()
    plt.grid()
    plt.savefig("/Users/roberto/code/speed-from-image/evaluation/validation-performance.png")

# RMSE comparisons
print("Average RMSE (over entire subset)")
print("Validation set:")
cumulative_RMSE = 0

for i in range(len(val_loader)):
    img = val_loader.dataset[i]['image'].unsqueeze_(0).unsqueeze_(0)
    speed_labels = val_loader.dataset[i]['speeds'].numpy()
    speed_estimate = model(img).detach().numpy()
    RMSE = np.sqrt(np.mean(np.square(speed_labels - speed_estimate) / len(speed_labels)))
    cumulative_RMSE += RMSE
print(cumulative_RMSE / len(val_loader))

cumulative_RMSE = 0
# Not sure if this is the best way to do this, but otherwise the batch size will be what it was in training
# and I don't know what that causes this loop to do.
train_loader = DataLoader(train_dataset, batch_size=1,
                          shuffle=False, num_workers=1)
print("Training set:")
for i in range(len(train_loader)):
    img = train_loader.dataset[i]['image'].unsqueeze_(0).unsqueeze_(0)
    speed_labels = train_loader.dataset[i]['speeds'].numpy()
    speed_estimate = model(img).detach().numpy()
    RMSE = np.sqrt(np.mean(np.square(speed_labels - speed_estimate) / len(speed_labels)))
    cumulative_RMSE += RMSE
#     print(RMSE)
print(cumulative_RMSE / len(train_loader))
