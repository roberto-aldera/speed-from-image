from torch.utils.data import DataLoader
from torchvision import transforms
import torch

import matplotlib.pyplot as plt

from dataset_loader import TunnelDataset, ToTensor
from network import Net

train_dataset = TunnelDataset(root_dir='/Users/roberto/code/speed-from-image/images/',
                              data_subset_type="training",
                              transform=transforms.Compose([ToTensor()]))
train_loader = DataLoader(train_dataset, batch_size=1,
                          shuffle=False, num_workers=1)
img_idx = 0
img = train_loader.dataset[img_idx]['image'].unsqueeze_(0).unsqueeze_(0)
print(img.shape)
speed_labels = train_loader.dataset[img_idx]['speeds']

model = Net()
model.load_state_dict(torch.load("/Users/roberto/code/speed-from-image/models/myModel.pt"))

speed_estimate = model(img).detach().numpy()
plt.plot(speed_labels, label='Ground truth')
plt.plot(speed_estimate[0], label='Prediction')
plt.xlabel('Index')
plt.ylabel('Speed (or width)')
plt.legend()
plt.grid()
print(speed_estimate)
plt.show()