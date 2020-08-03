import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import settings


class TunnelDataset(Dataset):
    """Tunnel dataset."""

    def __init__(self, root_dir, data_subset_type, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            data_subset_type (string): Specify if the required data is "training", "validation", or "test"
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.speeds_frame = pd.read_csv(root_dir + data_subset_type + "/" + data_subset_type + "_speed_labels.csv",
                                        header=None)
        self.root_dir = root_dir + data_subset_type + "/"
        self.data_subset_type = data_subset_type
        self.transform = transform

    def __len__(self):
        return len(self.speeds_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.root_dir + self.data_subset_type + "_" + str(idx) + ".png"
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


class Normalise(object):
    """Perform normalisation."""

    def __call__(self, sample):
        image, speeds = sample['image'], sample['speeds']
        mean = settings.TOY_SPEED_MEAN
        std_dev = settings.TOY_SPEED_STD_DEV
        scaled_speeds = (speeds - mean) / std_dev
        return {'image': image,
                'speeds': scaled_speeds}


def main():
    # Define a main loop to run and show some example data if this script is run as main
    tunnel_dataset = TunnelDataset(
        root_dir=settings.SIM_IMAGE_DIR,
        data_subset_type=settings.TRAIN_SUBSET)
    tunnels_idx = 0
    tunnel = tunnel_dataset[tunnels_idx]

    print(tunnels_idx, tunnel['image'].shape, tunnel['speeds'].shape)

    # plt.figure(figsize=(1, 1))
    # plt.imshow(tunnel['image'], cmap='gray', vmin=0, vmax=255)
    plt.figure(figsize=(10, 5))
    plt.plot(tunnel['speeds'], '.-')
    plt.ylim(-1, 20)
    plt.grid()
    plt.savefig("/workspace/Desktop/tmp.png")
    plt.close()


if __name__ == "__main__":
    main()
