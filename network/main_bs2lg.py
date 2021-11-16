import os

import function_training
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import json

import numpy as np
import tensorflow as tf

from dataLoader import AnyDataset
import function_training
from config import MLP_model,hyperparameters, args, CNN_model


def main(numBnads=100, bands_type="spin up"):
    # step 0. Check GPUs available:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available:", len(gpus))
    # set device
    device = tf.device('/GPU:0') if len(gpus) != 0 else tf.device('/CPU:0')

    # step 1. prepare dataset
    def json2inputlabel(data_json, bands_type="spin up"):
        _bands_type = {"spin up": "spin_up_bands",
                       "spin down": "spin_down_bands",
                       "soc": "soc_bands"}
        data_input_np = np.array(data_json[_bands_type[bands_type]]) # 100 x 400
        # data_input_np = np.array(data_json[_bands_type[bands_type]]).flatten().T # 40000x1
        data_label_np = np.array([data_json["new_label"]])
        return data_input_np, data_label_np

    train_dataset = AnyDataset("list/actual/train_set.txt", json2inputlabel, args["load"]["numClasses"],bands_type, training=True)
    test_dataset = AnyDataset("list/actual/test_set.txt", json2inputlabel, args["load"]["numClasses"],bands_type, training=False)
    train_loader = tf.data.Dataset.from_tensor_slices((train_dataset.data_inputs, train_dataset.data_labels)).shuffle(train_dataset.len).batch(hyperparameters['batch_size'])
    test_loader = tf.data.Dataset.from_tensor_slices((test_dataset.data_inputs,test_dataset.data_labels)).shuffle(test_dataset.len).batch(hyperparameters['batch_size'])

    # step 2. build model
    model = tf.keras.Sequential(MLP_model)

    # step 3. define loss
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # # step 4. Model compile
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=hyperparameters['learning_rate'],
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

    # step 5. select metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # step 6. train & evaluate & save model
    losses_train, accuracy_train, losses_test, accuracy_test = function_training.train_it(device, model, "./state_dicts/",
                                                                                          hyperparameters['epochs'],train_loader, test_loader, loss_fn, optimizer,
                                                                                          train_loss, train_accuracy, test_loss, test_accuracy,
                                                                                          eval_num_epochs=1, numClasss=args["load"]["numClasses"])

    # step 7 e

    pass


def testGpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    print("Num GPUS Available:", len(tf.config.experimental.list_physical_devices('GPU')))


if __name__ == '__main__':
    main()