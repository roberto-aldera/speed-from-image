import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import settings


class RadarDataset(Dataset):
    """Radar dataset."""

    def __init__(self, root_dir, data_subset_type, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            data_subset_type (string): Specify if the required data is "training", "validation", or "test"
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x_vals_frame = pd.read_csv(root_dir + data_subset_type + "/" + data_subset_type + "_x_vals_labels.csv",
                                        header=None)
        self.root_dir = root_dir + data_subset_type + "/"
        self.data_subset_type = data_subset_type
        self.transform = transform

    def __len__(self):
        return len(self.x_vals_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.root_dir + self.data_subset_type + "_" + str(idx) + ".png"
        image = np.array(Image.open(img_name))
        x_vals = np.array(self.x_vals_frame.iloc[idx])
        sample = {'image': image, 'x_vals': x_vals}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, x_vals = sample['image'], sample['x_vals']

        return {'image': torch.from_numpy(image),
                'x_vals': torch.from_numpy(x_vals)}


class Normalise(object):
    """Perform normalisation."""

    def __call__(self, sample):
        image, x_vals = sample['image'], sample['x_vals']
        mean = settings.ODOMETRY_SPEED_MEAN
        std_dev = settings.ODOMETRY_SPEED_STD_DEV
        scaled_x_vals = (x_vals - mean) / std_dev
        return {'image': image,
                'x_vals': scaled_x_vals}


def main():
    # Define a main loop to run and show some example data if this script is run as main
    radar_dataset = RadarDataset(
        root_dir=settings.RADAR_IMAGE_DIR,
        data_subset_type=settings.TRAIN_SUBSET)
    idx = 0
    radar_scan = radar_dataset[idx]

    print(idx, radar_scan['image'].shape, radar_scan['x_vals'].shape)

    # plt.figure(figsize=(1, 1))
    # plt.imshow(radar_scan['image'], cmap='gray', vmin=0, vmax=255)
    plt.figure(figsize=(10, 5))
    plt.plot(radar_scan['x_vals'], '.-')
    # plt.ylim(-1, 5)
    plt.grid()
    plt.savefig("/workspace/Desktop/radar-speeds-0.png")
    plt.close()


if __name__ == "__main__":
    main()
