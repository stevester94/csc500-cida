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