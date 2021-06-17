#! /usr/bin/env python3

import sys

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np

# from steves_utils import utils
# from torch_dataset_accessor.torch_windowed_shuffled_dataset_accessor import get_torch_windowed_shuffled_datasets
# from steves_utils.ORACLE.windowed_shuffled_dataset_accessor import Windowed_Shuffled_Dataset_Factory
# from steves_utils.ORACLE.utils import ALL_SERIAL_NUMBERS

torch.set_default_dtype(torch.float64)

# This is still wrong though
# def tf_categorical_crossentropy(y, y_hat, from_logits=False, axis=-1):
#     _EPSILON = 10e-8
#     y_hat = y_hat / tf.reduce_sum(y_hat, axis, True)



#     # Compute cross entropy from probabilities.
#     epsilon_ = tf.constant(_EPSILON, tf.float64)
#     y_hat = tf.clip_by_value(y_hat, epsilon_, 1. - epsilon_)
#     return tf.math.reduce_mean(-tf.reduce_sum(y * tf.math.log(y_hat), axis))

# Wait ok this is assuming a one hot encoding for the y
def steves_categorical_crossentropy(y_hat, y):
    epsilon = 10e-8
    y_hat = y_hat / torch.sum(y_hat, dim=-1, keepdim=True)
    y_hat = torch.clip(y_hat, epsilon, 1. - epsilon)
    return torch.mean(- torch.sum(y * torch.log(y_hat), dim=-1))


if __name__ == "__main__":
    import unittest

    NUM_RAND_TESTS = 10000
    MAX_NUM_CLASSES = 30
    MAX_BATCH_SIZE = 2000

    rng = np.random.default_rng(1337)

    cce = tf.keras.losses.CategoricalCrossentropy()

    class Test_Loss_Functions(unittest.TestCase):
        def test_equivalence(self):
            for i in range(NUM_RAND_TESTS):
                BATCH_SIZE = rng.integers(low=1, high=MAX_BATCH_SIZE, size=1)[0]
                NUM_CLASSES = rng.integers(low=2, high=MAX_NUM_CLASSES, size=1)[0]

                print("A")
                y_hat = torch.randn(BATCH_SIZE, NUM_CLASSES)
                y_hat = torch.softmax(y_hat, dim=1).numpy()
            
                y = torch.zeros([BATCH_SIZE, NUM_CLASSES]).numpy()
                for i in y:
                    rand_index = rng.integers(low=0, high=NUM_CLASSES-1, size=1)[0]
                    i[rand_index] = 1
                
                mine = steves_categorical_crossentropy(
                    y_hat=torch.tensor(y_hat),
                    y=torch.tensor(y)
                ).numpy()



                theirs = cce(
                    y_true=y,
                    y_pred=y_hat
                ).numpy()

                self.assertAlmostEqual(mine, theirs, places=5)

                # print(mine)
                # print(theirs)
                

                # print(y_hat)
    unittest.main()

# y = np.array(
#     [
#         [0,1,0],
#         [0,1,0],
#     ]
# )

# y_hat = np.array(
#     [
#         [0.0,0.0,1.0],
#         [0.0,1.0,0.0]
#     ]
# )

# print(
#     "tf_categorical_crossentropy:",
#     tf_categorical_crossentropy(
#         y_hat=torch.tensor(y_hat),
#         y=torch.tensor(y)
#     ).numpy()
# )

# print(
#     "steves_categorical_crossentropy:",
#     steves_categorical_crossentropy(
#         y_hat=torch.tensor(y_hat),
#         y=torch.tensor(y)
#     ).numpy()
# )



# cce = tf.keras.losses.CategoricalCrossentropy()
# print(
#     "tf.keras.losses.CategoricalCrossentropy:",
#     cce(
#         y_true=y,
#         y_pred=y_hat
#     ).numpy()
# )

sys.exit(0)

criterion = nn.CrossEntropyLoss()
print(
    "nn.CrossEntropyLoss:",
    criterion(
        torch.tensor([[0.0, 1.0, 0.0]]),
        torch.tensor([2])
    ).numpy()
)


criterion = nn.NLLLoss()
print(
    "nn.NLLLoss:",
    criterion(
        torch.tensor([[0.0, 1.0, 0.0]]),
        torch.tensor([2])
    ).numpy()
)