#! /usr/bin/env python3

import sys
from torch.utils.data import DataLoader, Dataset
import numpy as np

from steves_utils.ORACLE.windowed_shuffled_dataset_accessor import Windowed_Shuffled_Dataset_Factory

class Torch_Windowed_Shuffled_Dataset_Factory(Dataset):
    def __init__(self, ds):
        self.ds = ds.unbatch().as_numpy_iterator()

        # (IQ, serial_number_id, distance_feet)
        self.IQ = []
        self.serial_number_id = []
        self.distance_feet = []

        for i in ds.as_numpy_iterator():
            self.IQ.extend(
                np.split(i["IQ"], i["IQ"].shape[0])
            )
            self.serial_number_id.extend(
                np.split(i["serial_number_id"], i["serial_number_id"].shape[0])
            )
            self.distance_feet.extend(
                np.split(i["distance_feet"], i["distance_feet"].shape[0])
            )
            # self.data.append(
            #     (
            #         i["IQ"],
            #         i["serial_number_id"],
            #         i["distance_feet"],
            #     )
            # )

        assert(len(self.IQ) == len(self.serial_number_id))
        assert(len(self.IQ) == len(self.distance_feet))

    def __getitem__(self, index):
        return (
            self.IQ[index],
            self.serial_number_id[index],
            self.distance_feet[index],
        )

    def __len__(self):
        return len(self.IQ)

if __name__ == "__main__":
    path = "/mnt/wd500GB/CSC500/csc500-super-repo/datasets/automated_windower/windowed_EachDevice-200k_batch-100_stride-10_distances-2"
    datasets = Windowed_Shuffled_Dataset_Factory(path)

    train_ds = datasets["train_ds"]
    val_ds = datasets["val_ds"]
    test_ds = datasets["test_ds"]

    dsf = Torch_Windowed_Shuffled_Dataset_Factory(train_ds.take(1000))

    dataloader = DataLoader(
        dataset=dsf,
        shuffle=False,
        # batch_size=None # Disable batching
        batch_size=None
    )



    for i in dataloader:
        # print(i)
        # sys.exit(1)

        # print(i["run"])
        # print(i["distance_feet"])
        # print(i["serial_number_id"])
        # print(i["run"])
        # print(i["run"])

        pass
