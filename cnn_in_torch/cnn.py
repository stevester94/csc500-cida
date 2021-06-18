#! /usr/bin/env python3
import sys

import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tensorflow as tf

from steves_utils import utils
from torch_dataset_accessor.torch_windowed_shuffled_dataset_accessor import get_torch_windowed_shuffled_datasets
from steves_utils.ORACLE.windowed_shuffled_dataset_accessor import Windowed_Shuffled_Dataset_Factory
from steves_utils.ORACLE.utils import ALL_SERIAL_NUMBERS
from steves_loss import steves_categorical_crossentropy

torch.set_default_dtype(torch.float64)

# loss = nn.CrossEntropyLoss()
# input = torch.tensor(
#     np.array([[0,1.00]]),
#     requires_grad=True
# )

# target = torch.from_numpy(
#     np.array([1])
# )
# output = loss(input, target)

# print(output)

# output.backward()

# sys.exit(1)

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
#         x = self.f3(cx)
#         return x

# class CNNModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_channels=2, out_channels=50, kernel_size=7, stride=1)
#         self.pool = nn.MaxPool1d(2, 2)
#         self.conv2 = nn.Conv1d(in_channels=50, out_channels=50, kernel_size=7, stride=1)
#         self.fc1 = nn.Linear(1350, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 16)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         # print(x.shape)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# def tf_loss(y_hat_np, y_np):
#     _EPSILON = 10e-8

#     def _to_tensor(x, dtype):
#         """Convert the input `x` to a tensor of type `dtype`.

#         Args:
#             x: An object to be converted (numpy array, list, tensors).
#             dtype: The destination type.

#         Returns:
#             A tensor.
#         """

#         # Ripped from keras/backend.py, line 909
#         return tf.convert_to_tensor(x, dtype=dtype)


#     output = tf.convert_to_tensor(y_hat_np)
#     target = tf.convert_to_tensor(y_np)
    

#     # Ripped from https://github.com/keras-team/keras/blob/85fe6427a50f6def7777413a4bc8b286f7ab8f8d/keras/backend/tensorflow_backend.py#L2729

#     # scale preds so that the class probas of each sample sum to 1
#     output /= tf.math.reduce_sum(output,
#                             reduction_indices=len(output.get_shape()) - 1,
#                             keep_dims=True
#                 )
#     # manual computation of crossentropy
#     epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
#     output = tf.clip_by_value(output, epsilon, 1. - epsilon)
#     return - tf.reduce_sum(target * tf.math.log(output),
#                             reduction_indices=len(output.get_shape()) - 1)




class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()

        self.feature.add_module('f_conv1', nn.Conv1d(in_channels=2, out_channels=50, kernel_size=7, stride=1))
        self.feature.add_module('f_relu2', nn.ReLU(False))
        self.feature.add_module('f_conv2', nn.Conv1d(in_channels=50, out_channels=50, kernel_size=7, stride=2))
        self.feature.add_module('f_relu2', nn.ReLU(False))
        self.feature.add_module('f_drop1', nn.Dropout())
        self.feature.add_module('f_drop1', nn.Flatten())

        self.feature.add_module('c_fc1', nn.Linear(50 * 58, 256))
        self.feature.add_module('c_relu1', nn.ReLU(False))
        self.feature.add_module('c_drop1', nn.Dropout())
        self.feature.add_module('c_fc2', nn.Linear(256, 80))
        self.feature.add_module('c_relu2', nn.ReLU(False))
        self.feature.add_module('c_drop1', nn.Dropout())
        self.feature.add_module('c_fc3', nn.Linear(80, 16))
        # self.feature.add_module('c_softmax', nn.Softmax(dim=1))

    def forward(self, input_data):
        # print("input_data:", input_data.shape)
        # x = torch.ones(1024, 2, 128)
        # print(self.feature(x).shape)
        # import sys
        # sys.exit(1)

        class_output = self.feature(input_data)

        return class_output

source_distance = 50


DESIRED_BATCH_SIZE = 512
LEARNING_RATE = 0.0001
RANGE = len(ALL_SERIAL_NUMBERS)
ORIGINAL_BATCH_SIZE = 100



def apply_dataset_pipeline(datasets):
    """
    Apply the appropriate dataset pipeline to the datasets returned from the Windowed_Shuffled_Dataset_Factory
    """
    train_ds = datasets["train_ds"]
    val_ds = datasets["val_ds"]
    test_ds = datasets["test_ds"]

    # train_ds = train_ds.map(
    #     lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
    #     num_parallel_calls=tf.data.AUTOTUNE,
    #     deterministic=True
    # )

    # val_ds = val_ds.map(
    #     lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
    #     num_parallel_calls=tf.data.AUTOTUNE,
    #     deterministic=True
    # )

    # test_ds = test_ds.map(
    #     lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
    #     num_parallel_calls=tf.data.AUTOTUNE,
    #     deterministic=True
    # )

    train_ds = train_ds.map(
        lambda x: (x["IQ"], x["serial_number_id"]),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    val_ds = val_ds.map(
        lambda x: (x["IQ"], x["serial_number_id"]),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    test_ds = test_ds.map(
        lambda x: (x["IQ"], x["serial_number_id"]),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    train_ds = train_ds.unbatch()
    val_ds = val_ds.unbatch()
    test_ds = test_ds.unbatch()

    train_ds = train_ds.shuffle(100 * ORIGINAL_BATCH_SIZE, reshuffle_each_iteration=True)
    
    train_ds = train_ds.batch(DESIRED_BATCH_SIZE)
    val_ds  = val_ds.batch(DESIRED_BATCH_SIZE)
    test_ds = test_ds.batch(DESIRED_BATCH_SIZE)

    train_ds = train_ds.prefetch(100)
    val_ds   = val_ds.prefetch(100)
    test_ds  = test_ds.prefetch(100)

    return train_ds, val_ds, test_ds

def get_shuffled_and_windowed_from_pregen_ds():
    path = "{datasets_base_path}/automated_windower/windowed_EachDevice-200k_batch-100_stride-20_distances-{distance}".format(
        datasets_base_path=utils.get_datasets_base_path(), distance=source_distance
    )
    print(path)
    datasets = Windowed_Shuffled_Dataset_Factory(path)

    return apply_dataset_pipeline(datasets)

original_train_ds, original_val_ds, original_test_ds = get_shuffled_and_windowed_from_pregen_ds()

net = CNNModel().cuda()

# The input is expected to contain raw, unnormalized scores for each class.
criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()
# criterion = steves_categorical_crossentropy
# optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

for epoch in range(10):  # loop over the dataset multiple times

    original_train_iter = original_train_ds.as_numpy_iterator()

    running_loss = 0.0
    for i, data in enumerate(original_train_iter):
        # get the inputs; data is a list of [inputs, labels]
        # inputs, labels = data
        # inputs = torch.from_numpy(data["IQ"])
        # labels = torch.from_numpy(data["serial_number_id"]).long()

        inputs = torch.from_numpy(data[0])
        labels = torch.from_numpy(data[1]).long()

        # print(inputs)
        # print(labels)

        # sys.exit(1)

        inputs = inputs.cuda()
        labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        # loss = criterion(outputs, labels)
        # loss = torch.from_numpy(tf_loss(outputs.cpu().detach().numpy(), labels.cpu().numpy())).cuda()


        loss.backward()
        optimizer.step()

        # print(outputs)

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
    for data in original_test_ds:
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