#! /usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from steves_utils import utils
from torch_dataset_accessor.torch_windowed_shuffled_dataset_accessor import get_torch_windowed_shuffled_datasets


# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x



class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()

        self.feature.add_module('f_conv1', nn.Conv1d(in_channels=2, out_channels=50, kernel_size=7, stride=1))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv1d(in_channels=50, out_channels=50, kernel_size=7, stride=2))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_drop1', nn.Dropout())
        self.feature.add_module('f_drop1', nn.Flatten())

        self.feature.add_module('c_fc1', nn.Linear(50 * 58, 256))
        self.feature.add_module('c_relu1', nn.ReLU(True))
        self.feature.add_module('c_drop1', nn.Dropout())
        self.feature.add_module('c_fc2', nn.Linear(256, 80))
        self.feature.add_module('c_relu2', nn.ReLU(True))
        self.feature.add_module('c_fc3', nn.Linear(80, 16))
        self.feature.add_module('c_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data):
        # print("input_data:", input_data.shape)
        # x = torch.ones(1024, 2, 128)
        # print(self.feature(x).shape)
        # import sys
        # sys.exit(1)

        class_output = self.feature(input_data)

        return class_output

source_distance = 50

source_ds_path = "{datasets_base_path}/automated_windower/windowed_EachDevice-200k_batch-100_stride-20_distances-{distance}".format(
    datasets_base_path=utils.get_datasets_base_path(), distance=source_distance
)

datasets_source = get_torch_windowed_shuffled_datasets(source_ds_path)

train_ds_source = datasets_source["train_ds"]
test_ds_source = datasets_source["test_ds"]

train_dl = torch.utils.data.DataLoader(
    dataset=train_ds_source,
    batch_size=1024,
)


test_dl = torch.utils.data.DataLoader(
    dataset=test_ds_source,
    batch_size=1024,
)




net = CNNModel().cuda()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dl):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 5 == 0:    # print every 5 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 5))
            running_loss = 0.0

print('Finished Training')


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_dl:
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # calculate outputs by running images through the network
        outputs = net(inputs)

        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: %d %%' % (
    100 * correct / total))