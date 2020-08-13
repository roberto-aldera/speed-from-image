import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import settings


class MazeDataset(Dataset):
    """Maze dataset."""

    def __init__(self, root_dir, data_subset_type, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            data_subset_type (string): Specify if the required data is "training", "validation", or "test"
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir + data_subset_type + "/"
        self.data_subset_type = data_subset_type
        self.transform = transform

    def __len__(self):
        ratio = 0
        if self.data_subset_type == settings.TRAIN_SUBSET:
            ratio = settings.TRAIN_RATIO
        elif self.data_subset_type == settings.VAL_SUBSET:
            ratio = settings.VAL_RATIO
        elif self.data_subset_type == settings.TEST_SUBSET:
            ratio = settings.TEST_RATIO
        return int(ratio * settings.TOTAL_SAMPLES)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.root_dir + self.data_subset_type + "_" + str(idx) + ".png"
        image = np.array(Image.open(img_name))
        dx_data = pd.read_csv(
            self.root_dir + "/" + "speed_labels_" + self.data_subset_type + "_" + str(idx) + ".csv",
            header=None)
        sample = {'image': image, 'dx_data': dx_data}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, dx_data = sample['image'], sample['dx_data']
        return {'image': torch.from_numpy(image),
                'dx_data': torch.tensor(dx_data[0])}


class Normalise(object):
    """Perform normalisation."""

    def __call__(self, sample):
        image, dx_data = sample['image'], sample['dx_data']
        mean = settings.MAZE_SPEED_MEAN
        std_dev = settings.MAZE_SPEED_STD_DEV
        scaled_dx_data = (dx_data - mean) / std_dev
        return {'image': image,
                'dx_data': scaled_dx_data}


def main():
    # Define a main loop to run and show some example data if this script is run as main
    maze_dataset = MazeDataset(
        root_dir=settings.MAZE_IMAGE_DIR,
        data_subset_type=settings.TRAIN_SUBSET)

    for maze_idx in range(10):
        maze = maze_dataset[maze_idx]

        print(maze_idx, maze['image'].shape, maze['dx_data'].shape)

        # plt.figure(figsize=(1, 1))
        # plt.imshow(maze['image'], cmap='gray', vmin=0, vmax=255)
        plt.figure(figsize=(10, 5))
        plt.plot(maze['dx_data'], '.-')
        plt.ylim(0, 1)
        plt.grid()
        plt.savefig("%s%i%s" % ("/workspace/Desktop/speeds/idx-", maze_idx, ".png"))
        plt.close()


if __name__ == "__main__":
    main()
