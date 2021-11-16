
# TODO: DATA Loading
args = {
        "load": {
            "start": True,
            "numBands": 60,
            "load_from_dir": "../input_data/energy_separation03/",
            "group_type": "layergroup",
            "numClasses": 20,
            "num_upper_bound": 1000,
            "num_lower_bound": 0,
            "seed": None,
        }
}

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

# TODO:Deep learning model
MLP_model = [
    tf.keras.layers.Flatten(input_shape=(60, 400)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(10000, use_bias=True, bias_initializer='zeros',
                          kernel_regularizer=tf.keras.regularizers.L1(0.3),
                          activity_regularizer=tf.keras.regularizers.L2(0.3)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4096, use_bias=True, bias_initializer='zeros',
                          kernel_regularizer=tf.keras.regularizers.L1(0.7),
                          activity_regularizer=tf.keras.regularizers.L2(0.3)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1024, use_bias=True, bias_initializer='zeros',
                          kernel_regularizer=tf.keras.regularizers.L1(0.7),
                          activity_regularizer=tf.keras.regularizers.L2(0.3)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(256, use_bias=True, bias_initializer='zeros',
                          kernel_regularizer=tf.keras.regularizers.L1(0.7),
                          activity_regularizer=tf.keras.regularizers.L2(0.3)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(20),
]

CNN_model = [
    tf.keras.layers.ZeroPadding2D(padding=(3,3), data_format="channels_last", input_shape=(32,60,400)),
    tf.keras.layers.Conv2D(filters=10, kernel_size=(6, 6), padding='valid', strides=4, data_format="channels_last"),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format="channels_last"),
    tf.keras.layers.Conv2D(filters=20, kernel_size=(3, 6), padding='valid', strides=2, data_format="channels_last"),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format="channels_last"),
    tf.keras.layers.Flatten(data_format="channels_last"),
    tf.keras.layers.Dense(units=1024, use_bias=True, bias_initializer='zeros',
                          kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(units=256, use_bias=True, bias_initializer='zeros',
                          kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(units=20)
]


# TODO: Hyperparameters settings
hyperparameters = {
        "learning_rate": 0.00005,
        "epochs": 1000,
        "batch_size": 32,
}