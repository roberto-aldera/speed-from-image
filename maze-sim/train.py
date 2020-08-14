import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import time

import settings
from dataset_loader import MazeDataset, ToTensor, Normalise
from evaluate import do_quick_evaluation

start_time = time.time()

Path(settings.MAZE_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.MAZE_MODEL_DIR).mkdir(parents=True, exist_ok=True)

data_transform_for_training = transforms.Compose([ToTensor(), Normalise()])

train_dataset = MazeDataset(root_dir=settings.MAZE_IMAGE_DIR,
                            data_subset_type=settings.TRAIN_SUBSET,
                            transform=data_transform_for_training)

print("Training set size:", len(train_dataset))

sample = train_dataset[0]
print(0, sample['image'].size(), sample['dx_data'].size())

# Training starts here
train_loader = DataLoader(train_dataset, batch_size=16,
                          shuffle=True, num_workers=1)

torch.manual_seed(0)
learning_rate = 1e-4
net = settings.MODEL
criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

losses_over_epochs = []
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for batch_idx, sample_batched in enumerate(train_loader):
        inputs = sample_batched['image'].unsqueeze_(1)  # batch_size, channels, H, W
        labels = sample_batched['dx_data'].to(torch.float32)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # print("Inputs:", inputs.shape)
        # print("Labels:", labels.shape)
        # print("Outputs:", outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if batch_idx == len(train_loader) - 1:
            print("[%d, %5d] loss: %.3f" %
                  (epoch + 1, batch_idx + 1, running_loss / 1))
            losses_over_epochs.append(running_loss)
            running_loss = 0.0

    torch.save(net.state_dict(),
               "%s%s%s" % (settings.MAZE_MODEL_DIR, settings.ARCHITECTURE_TYPE, "_checkpoint.pt"))

    plt.figure(figsize=(15, 5))
    plt.plot(losses_over_epochs, '.-')
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("%s%s%s%s" % ("Loss after each epoch, model = ", settings.ARCHITECTURE_TYPE, ", batch size = ",
                            train_loader.batch_size))
    plt.savefig(settings.MAZE_RESULTS_DIR + "/training_loss.png")
    plt.close()

# Save final model with unique filename
torch.save(net.state_dict(), "%s%s%s%s%s" % (
    settings.MAZE_MODEL_DIR, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), "_", settings.ARCHITECTURE_TYPE,
    ".pt"))
print("Finished Training")
print("--- Training execution time: %s seconds ---" % (time.time() - start_time))

path_to_model = "%s%s%s" % (settings.MAZE_MODEL_DIR, settings.ARCHITECTURE_TYPE, "_checkpoint.pt")
do_quick_evaluation(model_path=path_to_model)
