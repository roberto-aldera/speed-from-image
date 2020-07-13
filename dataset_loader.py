import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torch

from torch.utils.data import Dataset


class TunnelDataset(Dataset):
    """Tunnel dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with targets.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.speeds_frame = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.speeds_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.root_dir + 'img-' + str(idx) + '.png'
        image = np.array(Image.open(img_name))
        speeds = np.array(self.speeds_frame.iloc[idx])
        sample = {'image': image, 'speeds': speeds}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, speeds = sample['image'], sample['speeds']

        return {'image': torch.from_numpy(image),
                'speeds': torch.from_numpy(speeds)}


def main():
    # Define a main loop to run and show some example data if this script is run as main
    tunnel_dataset = TunnelDataset(csv_file='/Users/roberto/code/speed-from-image/images/speed_labels.csv',
                                   root_dir='/Users/roberto/code/speed-from-image/images/')
    tunnels_idx = 0
    tunnel = tunnel_dataset[tunnels_idx]

    print(tunnels_idx, tunnel['image'].shape, tunnel['speeds'].shape)

    plt.figure(figsize=(1, 1))
    plt.imshow(tunnel['image'], cmap='gray', vmin=0, vmax=255)
    plt.figure(figsize=(5, 1))
    plt.plot(tunnel['speeds'])
    plt.show()


if __name__ == "__main__":
    main()
