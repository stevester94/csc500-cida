#! /usr/bin/env python3

import tensorflow as tf
from steves_utils.ORACLE.windowed_shuffled_dataset_accessor import Windowed_Shuffled_Dataset_Factory


def apply_dataset_pipeline(datasets, original_batch_size, desired_batch_size):
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
        lambda x: (x["IQ"], x["serial_number_id"], x["distance_feet"]),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    val_ds = val_ds.map(
        lambda x: (x["IQ"], x["serial_number_id"], x["distance_feet"]),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    test_ds = test_ds.map(
        lambda x: (x["IQ"], x["serial_number_id"], x["distance_feet"]),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    train_ds = train_ds.unbatch()
    val_ds = val_ds.unbatch()
    test_ds = test_ds.unbatch()

    train_ds = train_ds.shuffle(100 * original_batch_size, reshuffle_each_iteration=True)
    
    train_ds = train_ds.batch(desired_batch_size)
    val_ds  = val_ds.batch(desired_batch_size)
    test_ds = test_ds.batch(desired_batch_size)

    train_ds = train_ds.prefetch(100)
    val_ds   = val_ds.prefetch(100)
    test_ds  = test_ds.prefetch(100)

    return train_ds, val_ds, test_ds


def get_shuffled_and_windowed_from_pregen_ds(path, original_batch_size, desired_batch_size):
    datasets = Windowed_Shuffled_Dataset_Factory(path)

    return apply_dataset_pipeline(datasets, original_batch_size, desired_batch_size)




if __name__ == "__main__":
    from steves_utils import utils
    import timeit

    ORIGINAL_BATCH_SIZE = 100

    ds_path = "{datasets_base_path}/automated_windower/windowed_EachDevice-200k_batch-100_stride-20_distances-{distance}".format(
        datasets_base_path=utils.get_datasets_base_path(), distance="32"
    )   

    num_items = 0
    def get_all_items(ds, as_numpy):
        global num_items
        num_items = 0

        if as_numpy:
            for x in ds.as_numpy_iterator():
                num_items += x[0].shape[0]
        else:
            for x in ds:
                num_items += x[0].shape[0]
    


    for batch_size in [32, 64, 128, 256, 512, 1024]:
        train_ds_source, val_ds_source, test_ds_source = get_shuffled_and_windowed_from_pregen_ds(ds_path, ORIGINAL_BATCH_SIZE, batch_size)

        for i in range(3):
            t = timeit.timeit(lambda: get_all_items(train_ds_source, as_numpy=False), number=1)
            print("Examples per second from vanilla TF dataset with batch size {}: {:.2f}".format(batch_size, num_items/t))

        for i in range(3):
            t = timeit.timeit(lambda: get_all_items(train_ds_source, as_numpy=True), number=1)
            print("Examples per second from TF dataset as numpy iterator with batch size {}: {:.2f}".format(batch_size, num_items/t))