#! /usr/bin/env python3

import unittest
import numpy as np
import sys

from torch_dataset_accessor.torch_windowed_shuffled_dataset_accessor import get_torch_windowed_shuffled_datasets
from steves_utils.ORACLE.windowed_shuffled_dataset_accessor import Windowed_Shuffled_Dataset_Factory

def iq_and_serial_tuple_to_hash(t):
    # I have no idea why but the original hash method was not giving reproducible hashes (even when setting PYTHONSEED)
    # return hash(t[0].data.tobytes()) + hash(t[1].data.tobytes())
    return sum(t[0][0]) + sum(t[0][1]) + t[1]
        

class Test_Datasets_Equivalent(unittest.TestCase):
    def test_equivalence(self):
        path = "/mnt/wd500GB/CSC500/csc500-super-repo/datasets/automated_windower/windowed_EachDevice-200k_batch-100_stride-10_distances-2"

        # original_datasets = Windowed_Shuffled_Dataset_Factory(path)
        # original_train_ds = original_datasets["train_ds"].unbatch().take(1).as_numpy_iterator()
        # original_val_ds = original_datasets["val_ds"].unbatch().take(1).as_numpy_iterator()
        # original_test_ds = original_datasets["test_ds"].unbatch().take(1).as_numpy_iterator()

        # torch_datasets = get_torch_windowed_shuffled_datasets(path, take=1)
        # torch_train_ds = torch_datasets["train_ds"]
        # torch_val_ds = torch_datasets["val_ds"]
        # torch_test_ds = torch_datasets["test_ds"]

        # original_datasets = Windowed_Shuffled_Dataset_Factory(path, reshuffle_train_each_iteration=False)
        # torch_datasets = get_torch_windowed_shuffled_datasets(path, reshuffle_train_each_iteration=False)

        original_datasets = Windowed_Shuffled_Dataset_Factory(path)
        torch_datasets = get_torch_windowed_shuffled_datasets(path)

        torch_train_ds = torch_datasets["train_ds"]
        torch_val_ds = torch_datasets["val_ds"]
        torch_test_ds = torch_datasets["test_ds"]

        # for i in original_train_ds:
        #     print(i["IQ"].shape)
        
        # for i in torch_train_ds:
        #     print(i)

        # import sys
        # sys.exit(1)
        


        for _ in range(3):
            original_train_ds = original_datasets["train_ds"].unbatch().as_numpy_iterator()
            original_val_ds = original_datasets["val_ds"].unbatch().as_numpy_iterator()
            original_test_ds = original_datasets["test_ds"].unbatch().as_numpy_iterator()

            for torch_ds, original_ds in [(torch_train_ds, original_train_ds), (torch_val_ds, original_val_ds), (torch_test_ds, original_test_ds)]:
                torch_l = []
                original_l = []

                for i in torch_ds:
                    torch_l.append(iq_and_serial_tuple_to_hash(i))

                for i in original_ds:
                    original_l.append(iq_and_serial_tuple_to_hash((i["IQ"], i["serial_number_id"])))

                self.assertEqual(len(original_l), len(torch_l))

                torch_s = set(torch_l)
                original_s = set(original_l)

                # print("torch:",len(torch_s), len(torch_l))
                # print("original:",len(original_s), len(original_l))

                # self.assertEqual(len(torch_s), len(torch_l))
                # self.assertEqual(len(original_s), len(original_l))
            
                self.assertEqual(torch_s, original_s)

                print("Done with pass")



if __name__ == "__main__":
    unittest.main()
    sys.exit(0)


    path = "/mnt/wd500GB/CSC500/csc500-super-repo/datasets/automated_windower/windowed_EachDevice-200k_batch-100_stride-10_distances-2"

    # original_datasets = Windowed_Shuffled_Dataset_Factory(path)
    # original_train_ds = original_datasets["train_ds"].unbatch().take(1).as_numpy_iterator()
    # original_val_ds = original_datasets["val_ds"].unbatch().take(1).as_numpy_iterator()
    # original_test_ds = original_datasets["test_ds"].unbatch().take(1).as_numpy_iterator()

    # torch_datasets = get_torch_windowed_shuffled_datasets(path, take=1)
    # torch_train_ds = torch_datasets["train_ds"]
    # torch_val_ds = torch_datasets["val_ds"]
    # torch_test_ds = torch_datasets["test_ds"]

    original_datasets = Windowed_Shuffled_Dataset_Factory(path, reshuffle_train_each_iteration=False)


    torch_datasets = get_torch_windowed_shuffled_datasets(path, reshuffle_train_each_iteration=False, take=100)
    torch_train_ds = torch_datasets["train_ds"]
    torch_val_ds = torch_datasets["val_ds"]
    torch_test_ds = torch_datasets["test_ds"]

    # for i in original_train_ds:
    #     print(i["IQ"].shape)
    
    # for i in torch_train_ds:
    #     print(i)

    # import sys
    # sys.exit(1)
    


    original_train_ds = original_datasets["train_ds"].unbatch().take(100).as_numpy_iterator()
    original_val_ds = original_datasets["val_ds"].unbatch().take(100).as_numpy_iterator()
    original_test_ds = original_datasets["test_ds"].unbatch().take(100).as_numpy_iterator()

    for torch_ds, original_ds in [(torch_train_ds, original_train_ds), (torch_val_ds, original_val_ds), (torch_test_ds, original_test_ds)]:
        torch_l = []
        original_l = []

        torch_iter = iter(torch_ds)
        orig_iter  = original_ds

        while True:
            try:
                t = next(torch_iter)
                o = next(orig_iter)
            except:
                print("Iterator Exhausted")
                break

            t_iq = t[0]
            o_iq = o["IQ"]

            t_serial = t[1]
            o_serial = o["serial_number_id"]

            if not np.array_equal(t_iq, o_iq):
                print("IQ NOT EQUAL")
                sys.exit(1)

            if t_serial != o_serial:
                print("serial NOT EQUAL")
                sys.exit(1)

            t_hash = iq_and_serial_tuple_to_hash( 
                (t_iq, t_serial)
            )

            o_hash = iq_and_serial_tuple_to_hash( 
                (o_iq, o_serial)
            )

            print(t_hash)
            print(o_hash)


            sys.exit(1)

        # for i in torch_ds:
        #     torch_l.append(iq_and_serial_tuple_to_hash(i))

        # for i in original_ds:
        #     original_l.append(iq_and_serial_tuple_to_hash((i["IQ"], i["serial_number_id"])))

        # # self.assertEqual(len(original_l), len(torch_l))

        # torch_s = set(torch_l)
        # original_s = set(original_l)

        # print("torch:",len(torch_s), len(torch_l))
        # print("original:",len(original_s), len(original_l))

        # # self.assertEqual(len(torch_s), len(torch_l))
        # # self.assertEqual(len(original_s), len(original_l))
    
        # # self.assertEqual(torch_s, original_s)

        # print("Done with pass")