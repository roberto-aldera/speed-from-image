import torch.nn.functional as func
import settings
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from dataset_loader import MazeDataset, CollateFn, ToTensor, Normalise
from lenet import LeNet
from resnet import resnet18


class MyLightningTemplateModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return self.net(x)

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


class LeNetLightningTemplateModel(MyLightningTemplateModel):
    def __init__(self):
        super().__init__()
        self.net = LeNet()


class ResNetLightningTemplateModel(MyLightningTemplateModel):
    def __init__(self):
        super().__init__()
        self.net = resnet18()
