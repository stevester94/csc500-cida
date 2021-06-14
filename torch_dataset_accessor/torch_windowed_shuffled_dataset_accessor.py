#! /usr/bin/env python3

import sys
from torch.utils.data import DataLoader, Dataset, IterableDataset
import numpy as np

from steves_utils.ORACLE.windowed_shuffled_dataset_accessor import Windowed_Shuffled_Dataset_Factory

      
def get_torch_windowed_shuffled_datasets(path, take=None):
    datasets = Windowed_Shuffled_Dataset_Factory(path)

    train_ds = datasets["train_ds"]
    val_ds = datasets["val_ds"]
    test_ds = datasets["test_ds"]

    # return {
    #     "train_ds": ORACLE_TF_DS_to_torch_DS(train_ds),
    #     "val_ds": ORACLE_TF_DS_to_torch_DS(val_ds),
    #     "test_ds": ORACLE_TF_DS_to_torch_DS(test_ds),
    # }

    return {
        "train_ds": ORACLE_TF_DS_to_Iterable_Torch_DS(train_ds, take=take),
        "val_ds": ORACLE_TF_DS_to_Iterable_Torch_DS(val_ds, take=take),
        "test_ds": ORACLE_TF_DS_to_Iterable_Torch_DS(test_ds, take=take),
    }

class ORACLE_TF_DS_to_torch_DS(Dataset):
    """
    Reads the entire dataset int memory (note this is untennable for the real windowed datasets)
    """
    def __init__(self, ds):
        self.ds = ds.as_numpy_iterator()

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

        self.len = 0
        for i in ds.unbatch().batch(100000):
            self.len += i["IQ"].shape[0]

        assert(len(self.IQ) == len(self.serial_number_id))
        assert(len(self.IQ) == len(self.distance_feet))

    def __getitem__(self, index):
        return (
            self.IQ[index],
            self.serial_number_id[index],
            self.distance_feet[index],
        )

    def __len__(self):
        return self.len

class ORACLE_TF_DS_to_Iterable_Torch_DS(IterableDataset):
    def __init__(self, ds, take=None):
        if take is not None:
            self.ds = ds.unbatch().take(take)
        else:
            self.ds = ds.unbatch()
        self.len = 0
        for i in self.ds.batch(100000):
            self.len += i["IQ"].shape[0]

    def __next__(self):
        # print("__next__")
        e = next(self.iter)

        return (
                e["IQ"].astype("float32"), 
                e["serial_number_id"].astype("long")
        )

    def __iter__(self):
        # print("__iter__")
        self.iter = self.ds.as_numpy_iterator()

        return self

    def __len__(self):
        return self.len





if __name__ == "__main__":
    datasets_source = get_torch_windowed_shuffled_datasets(
        "/mnt/wd500GB/CSC500/csc500-super-repo/datasets/automated_windower/windowed_EachDevice-200k_batch-100_stride-10_distances-2",
        take=1000
    )

    train_ds_source = datasets_source["train_ds"]

    print("One")
    count = 0
    for i in train_ds_source:
        count += 1
        print(count)

    print("Two")
    count = 0
    for i in train_ds_source:
        count += 1
        print(count)

    print("Three")
    count = 0
    for i in train_ds_source:
        count += 1
        print(count)