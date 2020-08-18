import torch.nn as nn
import torch.nn.functional as func
import settings
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from dataset_loader import MazeDataset, CollateFn, ToTensor, Normalise


class LeNet(pl.LightningModule):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_features=576, out_features=400)
        self.fc2 = nn.Linear(in_features=400, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=settings.MAX_ITERATIONS * settings.NUM_POSE_DIMS)

    def forward(self, x):
        x = x.unsqueeze_(1)
        # Max pooling over a (2, 2) window
        x = func.max_pool2d(func.relu(self.conv1(x.float())), (2, 2))
        # If the size is a square you can only specify a single number
        x = func.max_pool2d(func.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        # attempt at making this work for more than just dx (so dy, dth)
        x = x.view(-1, settings.NUM_POSE_DIMS, settings.MAX_ITERATIONS)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = func.mse_loss(y_hat, y.float())
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        return {'val_loss': func.mse_loss(y_hat, y.float())}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        return {'test_loss': func.mse_loss(y_hat, y.float())}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=settings.LEARNING_RATE)

    def train_dataloader(self):
        data_transform_for_training = transforms.Compose([ToTensor(), Normalise()])
        train_dataset = MazeDataset(root_dir=settings.MAZE_IMAGE_DIR,
                                    data_subset_type=settings.TRAIN_SUBSET,
                                    transform=data_transform_for_training)
        return DataLoader(train_dataset, batch_size=32,
                          shuffle=True, num_workers=4, collate_fn=CollateFn)

    def val_dataloader(self):
        data_transform_for_evaluation = transforms.Compose([ToTensor()])
        val_dataset = MazeDataset(root_dir=settings.MAZE_IMAGE_DIR,
                                  data_subset_type=settings.VAL_SUBSET,
                                  transform=data_transform_for_evaluation)
        return DataLoader(val_dataset, batch_size=32,
                          shuffle=False, num_workers=4, collate_fn=CollateFn)

    def test_dataloader(self):
        data_transform_for_evaluation = transforms.Compose([ToTensor()])
        test_dataset = MazeDataset(root_dir=settings.MAZE_IMAGE_DIR,
                                   data_subset_type=settings.TEST_SUBSET,
                                   transform=data_transform_for_evaluation)
        return DataLoader(test_dataset, batch_size=32,
                          shuffle=False, num_workers=4, collate_fn=CollateFn)
