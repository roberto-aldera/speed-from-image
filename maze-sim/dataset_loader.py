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
        pose_data = pd.read_csv(
            self.root_dir + "/" + "speed_labels_" + self.data_subset_type + "_" + str(idx) + ".csv",
            header=None)
        sample = {'image': image, 'pose_data': pose_data}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, pose_data = sample['image'], sample['pose_data']
        # print("Shape:", pose_data.shape)
        # print("Type:", type(pose_data))
        # print("pose_data: ", torch.tensor(pose_data.values))
        return {'image': torch.from_numpy(image),
                'pose_data': torch.tensor(pose_data.values)}


class Normalise(object):
    """Perform normalisation."""

    def __call__(self, sample):
        image, pose_data = sample['image'], sample['pose_data']
        mean = settings.MAZE_SPEED_MEAN
        std_dev = settings.MAZE_SPEED_STD_DEV
        scaled_pose_data = (pose_data - mean) / std_dev
        return {'image': image,
                'pose_data': scaled_pose_data}


def main():
    # Define a main loop to run and show some example data if this script is run as main
    maze_dataset = MazeDataset(
        root_dir=settings.MAZE_IMAGE_DIR,
        data_subset_type=settings.TRAIN_SUBSET)

    for maze_idx in range(2):
        maze = maze_dataset[maze_idx]
        print(maze_idx, maze['image'].shape, maze['pose_data'].shape)

        # plt.figure(figsize=(1, 1))
        # plt.imshow(maze['image'], cmap='gray', vmin=0, vmax=255)
        plt.figure(figsize=(10, 5))
        plt.plot(maze['pose_data'].iloc[0, :], '.-', label="dx")
        plt.plot(maze['pose_data'].iloc[1, :], '.-', label="dy")
        plt.plot(maze['pose_data'].iloc[2, :], '.-', label="dth")
        plt.ylim(-1, 1)
        plt.grid()
        plt.legend()
        plt.savefig("%s%i%s" % ("/workspace/Desktop/speeds/idx-", maze_idx, ".png"))
        plt.close()


if __name__ == "__main__":
    main()
