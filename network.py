import torch.nn as nn
import torch.nn.functional as func


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(3136, 1200)  # 6*6 from image dimension
        self.fc2 = nn.Linear(1200, 300)
        self.fc3 = nn.Linear(300, 64)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = func.max_pool2d(func.relu(self.conv1(x.float())), (2, 2))
        # If the size is a square you can only specify a single number
        x = func.max_pool2d(func.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# net = Net()
# print(net)
