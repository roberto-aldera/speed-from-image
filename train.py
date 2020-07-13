import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_loader import TunnelDataset, ToTensor
from network import Net

# Read in dataset and split into train, validate, and test data
full_dataset = TunnelDataset(csv_file='/Users/roberto/code/speed-from-image/images/speed_labels.csv',
                             root_dir='/Users/roberto/code/speed-from-image/images/',
                             transform=transforms.Compose([ToTensor()]))
train_ratio = 0.7
validate_ratio = 0.1
train_size = int(train_ratio * len(full_dataset))
val_size = int(validate_ratio * len(full_dataset))
test_size = len(full_dataset) - (train_size + val_size)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset,
                                                                         [train_size, val_size, test_size])
print('Training set size:', len(train_dataset))
print('Validation set size:', len(val_dataset))
print('Test set size:', len(test_dataset))

sample = train_dataset[0]
print(0, sample['image'].size(), sample['speeds'].size())

# Training starts here
train_loader = DataLoader(train_dataset, batch_size=16,
                          shuffle=True, num_workers=1)

torch.manual_seed(0)
learning_rate = 1e-3
net = Net()
criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

losses_over_epochs = []
for epoch in range(50):  # loop over the dataset multiple times
    running_loss = 0.0
    for batch_idx, sample_batched in enumerate(train_loader):
        inputs = sample_batched['image'].unsqueeze_(1)  # batch_size, channels, H, W
        labels = sample_batched['speeds'].to(torch.float32)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if batch_idx == len(train_loader) - 1:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 1))
            losses_over_epochs.append(running_loss)
            running_loss = 0.0
#             print(inputs.shape)
#             print(labels.shape)

print('Finished Training')
plt.figure(figsize=(15, 5))
plt.plot(losses_over_epochs, '.-')
plt.title("Loss after each epoch")
plt.show()