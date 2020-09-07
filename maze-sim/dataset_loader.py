import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
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
        subset_size = 0
        if self.data_subset_type == settings.TRAIN_SUBSET:
            subset_size = settings.TRAIN_SET_SIZE
        elif self.data_subset_type == settings.VAL_SUBSET:
            subset_size = settings.VAL_SET_SIZE
        elif self.data_subset_type == settings.TEST_SUBSET:
            subset_size = settings.TEST_SET_SIZE
        return subset_size

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


def CollateFn(batch):
    images = torch.stack([x['image'] for x in batch])
    poses = torch.stack([x['pose_data'] for x in batch])
    return images, poses


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, pose_data = sample['image'], sample['pose_data']
        return {'image': torch.from_numpy(image),
                'pose_data': torch.tensor(pose_data.values)}


class Normalise(object):
    """Perform normalisation."""

    def __call__(self, sample):
        image, pose_data = sample['image'], sample['pose_data']
        mean = settings.MAZE_SPEED_MEAN
        std_dev = settings.MAZE_SPEED_STD_DEV
        scaled_pose_data = np.transpose((np.transpose(pose_data) - mean) / std_dev)
        return {'image': image,
                'pose_data': scaled_pose_data}


def main():
    # Define a main loop to run and show some example data if this script is run as main
    data_transform_raw = transforms.Compose([ToTensor()])
    data_transform_scaled = transforms.Compose([ToTensor(), Normalise()])
    maze_dataset = MazeDataset(root_dir=settings.MAZE_IMAGE_DIR,
                               data_subset_type=settings.TRAIN_SUBSET,
                               transform=data_transform_raw)
    maze_elm_0 = maze_dataset[0]
    poses = np.array(maze_elm_0['pose_data'])

    plot_figures = False

    for maze_idx in range(1, len(maze_dataset)):
        maze = maze_dataset[maze_idx]
        # print(maze_idx, maze['image'].shape, maze['pose_data'].shape)
        poses = np.append(poses, maze['pose_data'], axis=1)

        if plot_figures:
            # plt.figure(figsize=(1, 1))
            # plt.imshow(maze['image'], cmap='gray', vmin=0, vmax=255)
            plt.figure(figsize=(10, 5))
            plt.plot(maze['pose_data'][0, :], '.-', label="dx")
            plt.plot(maze['pose_data'][1, :], '.-', label="dy")
            plt.plot(maze['pose_data'][2, :], '.-', label="dth")
            # plt.ylim(-1, 1)
            plt.grid()
            plt.legend()
            plt.savefig("%s%i%s" % ("/workspace/Desktop/speeds/idx-", maze_idx, ".png"))
            plt.close()
    print("shape:", poses.shape)
    print("Means:", np.mean(poses, axis=1))
    print("Std dev:", np.std(poses, axis=1))


if __name__ == "__main__":
    main()
