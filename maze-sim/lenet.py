import torch.nn as nn
import torch.nn.functional as func
import settings

dims = 2


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_features=576, out_features=400)
        self.fc2 = nn.Linear(in_features=400, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=settings.MAX_ITERATIONS * dims)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = func.max_pool2d(func.relu(self.conv1(x.float())), (2, 2))
        # If the size is a square you can only specify a single number
        x = func.max_pool2d(func.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        # attempt at making this work for more than just dx (so dy, dth)
        x = x.view(-1, dims, settings.MAX_ITERATIONS)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
