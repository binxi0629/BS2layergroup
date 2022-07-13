#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from matplotlib import pyplot as plt


# In[3]:


import os
import function_training
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import json
import datetime

import numpy as np
import tensorflow as tf
print(tf.__version__)
import keras_tuner as kt
print(kt.__version__)
from tensorboard.plugins.hparams import api as hp

from dataLoader import AnyDataset
import function_training
from config import args
# from config import MLP_model,hyperparameters, args, CNN_model

numBands=100
bands_type="spin up"


# In[4]:


# hyperparameters = {
# #         "learning_rate": 0.00005,
# #         "epochs": 100,
#         "batch_size": 32,
# }


# In[5]:


# step 0. Check GPUs available:
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available:", len(gpus))
# set device
device = tf.device('/GPU:0') if len(gpus) != 0 else tf.device('/CPU:0')


# In[6]:


# step 1. prepare dataset
def json2inputlabel(data_json, bands_type="spin up"):
    _bands_type = {"spin up": "spin_up_bands",
                   "spin down": "spin_down_bands",
                   "soc": "soc_bands"}
    data_input_np = np.array(data_json[_bands_type[bands_type]]) # 100 x 400
    # data_input_np = np.array(data_json[_bands_type[bands_type]]).flataten().T # 40000x1
    data_label_np = np.array([data_json["new_label"]])
    # data_label_np = np.array([data_json["layers_num"]])

    return data_input_np, data_label_np


#take data and set batch size here
train_dataset = AnyDataset("list/actual/train_set.txt", json2inputlabel, args["load"]["numClasses"],bands_type, training=True)
test_dataset = AnyDataset("list/actual/test_set.txt", json2inputlabel, args["load"]["numClasses"],bands_type, training=False)
# train_loader = tf.data.Dataset.from_tensor_slices((train_dataset.data_inputs, train_dataset.data_labels)).shuffle(train_dataset.len).batch(hyperparameters['batch_size'])
# test_loader = tf.data.Dataset.from_tensor_slices((test_dataset.data_inputs,test_dataset.data_labels)).shuffle(test_dataset.len).batch(hyperparameters['batch_size'])


# In[7]:


def model_builder(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(60, 100)))
    model.add(tf.keras.layers.LeakyReLU())
    
#     for i in range(hp.Int('layers', 1, 5)):
#         hppp = hp.Int(f'hidden_layer_{i}_units', 50, 100, step=10)
#         hp_activation = hp.Choice('act_' + str(i), ['relu', 'sigmoid'])
#         model.add(tf.keras.layers.Dense(units=hppp, activation=hp_activation))
  
    hp_L1 = 0.7
    L1_list = np.arange(0.1, 1, 0.1).tolist()
    hp_L1 = hp.Choice('L1 regularizer', values=L1_list)
    
    #TODO layer size tuning
    hp_layers = hp.Int('layers', 1, 4)
    for i in range(hp_layers):
#         hp_units = hp.Choice(f'hidden_layer_{i}_units', [5])
        hp_units = hp.Choice(f'hidden_layer_{i}_units', [5, 125, 625, 3125])
        model.add(tf.keras.layers.Dense(hp_units, use_bias=True, bias_initializer='zeros',
                          kernel_regularizer=tf.keras.regularizers.L1(hp_L1),
                          activity_regularizer=tf.keras.regularizers.L2(1-hp_L1)))
        model.add(tf.keras.layers.LeakyReLU())
    
    ## output layers
    model.add(tf.keras.layers.Dense(5))
    model.add(tf.keras.layers.LeakyReLU())

    # TO CHECK
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#     loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    # hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-5, 1e-6])
    # hp_decay_steps = hp.Choice('decay_steps', values=[1e3, 1e4])
    # hp_decay_rate = hp.Choice('decay_rate', values=[0.8, 0.85, 0.9, 0.95])

    #Small size testing 
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-5])
    hp_decay_steps = hp.Choice('decay_steps', values=[1e4])
    hp_decay_rate = hp.Choice('decay_rate', values=[0.9])
    
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=hp_learning_rate,
        decay_steps=hp_decay_steps,
        decay_rate=hp_decay_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

#     with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
#         hp.hparams_config(hparams=[hp_layers])
    
    model.compile(optimizer=optimizer,
                loss=loss_fn,
                metrics=['accuracy'])

    return model

# run parameter
log_dir = "logs/" + datetime.datetime.now().strftime("%m%d-%H%M")

# stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

hist_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    
    histogram_freq=0,
    update_freq="epoch", #epoch or batch?
    embeddings_freq=0,
    embeddings_metadata=None,
    write_graph=False,
    write_images=False,
    write_steps_per_second=False,
    profile_batch=0,   
)

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy', ###
                     max_epochs=300,
                     factor=3,
                     directory=log_dir,
#                      directory='./hp_output/',
                     project_name='BS2LN', overwrite=True)



print("log_dir:", log_dir)


# In[8]:


# Input data formatting
train_dataset_input = np.array(train_dataset.data_inputs)
test_dataset_input = np.array(test_dataset.data_inputs)
train_dataset_labels = np.array(train_dataset.data_labels).reshape(900,)
test_dataset_labels = np.array(test_dataset.data_labels).reshape(100,)


# In[ ]:


# tuner.search(train_dataset.data_inputs, test_dataset.data_inputs, epochs=50, validation_split=0.2, callbacks=[stop_early])

# tuner.search(train_dataset_input, train_dataset_labels, epochs=5, validation_split=0.2, callbacks=[stop_early])
# tuner.search(train_dataset_input, train_dataset_labels, epochs=300, validation_split=0.2, callbacks=[stop_early])
tuner.search(train_dataset_input, train_dataset_labels, epochs=5, validation_data=(test_dataset_input, test_dataset_labels), callbacks=[hist_callback], use_multiprocessing=True, verbose=1)
# tuner.search(train_dataset_input, train_dataset_labels, epochs=100, validation_data=(test_dataset_input, test_dataset_labels), callbacks=[stop_early], use_multiprocessing=True)
# best_model = tuner.get_best_models()[0]

# # Get the optimal hyperparameters
# best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]


# In[ ]:


# best_model = tuner.get_best_models()[0]
#tuner.get_best_hyperparameters(num_trials=1)[0]
# print(tuner.get_best_hyperparameters())
tuner.results_summary(num_trials=1)


# In[ ]:


tuner.search_space_summary()


# In[ ]:


# # MLP_model = [
# #     tf.keras.layers.Flatten(input_shape=(60, 100)),
# #     # tf.keras.layers.Flatten(input_shape=(60, 400)),
# #     tf.keras.layers.LeakyReLU(),

# #     tf.keras.layers.Dense(10000, use_bias=True, bias_initializer='zeros',
# #                           kernel_regularizer=tf.keras.regularizers.L1(0.7),
# #                           activity_regularizer=tf.keras.regularizers.L2(0.3)),
# #     tf.keras.layers.LeakyReLU(),
# #     tf.keras.layers.Dropout(0.3),

# #     tf.keras.layers.Dense(3125, use_bias=True, bias_initializer='zeros',
# #                           kernel_regularizer=tf.keras.regularizers.L1(0.7),
# #                           activity_regularizer=tf.keras.regularizers.L2(0.3)),
# #     tf.keras.layers.LeakyReLU(),
# #     tf.keras.layers.Dropout(0.3),


# #     tf.keras.layers.Dense(625, use_bias=True, bias_initializer='zeros',
# #                           kernel_regularizer=tf.keras.regularizers.L1(0.7),
# #                           activity_regularizer=tf.keras.regularizers.L2(0.3)),
# #     tf.keras.layers.LeakyReLU(),
# #     tf.keras.layers.Dropout(0.3),

# #     tf.keras.layers.Dense(125, use_bias=True, bias_initializer='zeros',
# #                           kernel_regularizer=tf.keras.regularizers.L1(0.7),
# #                           activity_regularizer=tf.keras.regularizers.L2(0.3)),
# #     tf.keras.layers.LeakyReLU(),


# #     tf.keras.layers.Dense(5),
# #     tf.keras.layers.LeakyReLU(),
# # #     tf.keras.layers.Softmax()
# # ]
# MLP_model = [
#     tf.keras.layers.Flatten(input_shape=(60, 100)),
#     # tf.keras.layers.Flatten(input_shape=(60, 400)),
#     tf.keras.layers.LeakyReLU(),

# #     tf.keras.layers.Dense(10000, use_bias=True, bias_initializer='zeros',
# #                           kernel_regularizer=tf.keras.regularizers.L1(0.7),
# #                           activity_regularizer=tf.keras.regularizers.L2(0.3)),
# #     tf.keras.layers.LeakyReLU(),
# #     tf.keras.layers.Dropout(0.3),

#     tf.keras.layers.Dense(3125, use_bias=True, bias_initializer='zeros',
#                           kernel_regularizer=tf.keras.regularizers.L1(0.7),
#                           activity_regularizer=tf.keras.regularizers.L2(0.3)),
#     tf.keras.layers.LeakyReLU(),
#     tf.keras.layers.Dropout(0.3),


#     tf.keras.layers.Dense(625, use_bias=True, bias_initializer='zeros',
#                           kernel_regularizer=tf.keras.regularizers.L1(0.7),
#                           activity_regularizer=tf.keras.regularizers.L2(0.3)),
#     tf.keras.layers.LeakyReLU(),
#     tf.keras.layers.Dropout(0.3),

#     tf.keras.layers.Dense(125, use_bias=True, bias_initializer='zeros',
#                           kernel_regularizer=tf.keras.regularizers.L1(0.7),
#                           activity_regularizer=tf.keras.regularizers.L2(0.3)),
#     tf.keras.layers.LeakyReLU(),


#     tf.keras.layers.Dense(5),
#     tf.keras.layers.LeakyReLU(),
# #     tf.keras.layers.Softmax()
# ]


# In[ ]:


# array = np.array(train_dataset.data_inputs)
# np.unique(array)


# In[ ]:


# x=810
# array[x:x+100]


# In[ ]:


# # step 2. build model
# model = tf.keras.Sequential(MLP_model)

# # step 3. define loss
# loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# # # step 4. Model compile
# lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=hyperparameters['learning_rate'],
#     decay_steps=10000,
#     decay_rate=0.9)
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

# # step 5. select metrics
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# In[ ]:


# # step 6. train & evaluate & save model
# losses_train, accuracy_train, losses_test, accuracy_test, model = function_training.train_it(device, model, "./state_dicts/",
#                                                                                       hyperparameters['epochs'],train_loader, test_loader, loss_fn, optimizer,
#                                                                                       train_loss, train_accuracy, test_loss, test_accuracy,
#                                                                                       eval_num_epochs=1, numClasss=args["load"]["numClasses"])


# In[ ]:


# for test_x, test_labels in test_loader:
#     print (f"model {model(test_x,training=False)}, actual {test_labels}")


# In[ ]:


# def plot_confusion_matrix(cm, class_names):
#     """
#     Returns a matplotlib figure containing the plotted confusion matrix.
    
#     Args:
#        cm (array, shape = [n, n]): a confusion matrix of integer classes
#        class_names (array, shape = [n]): String names of the integer classes
#     """
#     plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
#     plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    
#     figure = plt.figure(figsize=(8, 8), dpi=400)
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title("Confusion matrix")
# #     plt.colorbar()
#     tick_marks = np.arange(len(class_names))
#     plt.xticks(tick_marks, class_names)
# #     plt.xticks(tick_marks, class_names, rotation=45)
#     plt.yticks(tick_marks, class_names)


#     # Normalize the confusion matrix.
#     cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
#     # Use white text if squares are dark; otherwise black.
#     threshold = cm.max() / 2.
    
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             color = "white" if cm[i, j] > threshold else "black"
#             plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Prediction')
# #     return figure


# In[ ]:


# test_loader2 = tf.data.Dataset.from_tensor_slices((test_dataset.data_inputs,test_dataset.data_labels)).shuffle(train_dataset.len).batch(1000)

# # model2 = tf.keras.Sequential(MLP_model)

# for element, labels in test_loader2:
# #     print (element)
# #     print (labels)
# #     np.maximum([model(element, training=False)])
#     predict = np.argmax(model.predict(element), axis=-1)
#     label = np.array(labels).flatten()
# #     print (f"model {model.predict_classes(element)}, actual {labels}")
# #     model.predict_classes(element)
#     break


# In[ ]:


# cm = np.array(tf.math.confusion_matrix(labels=label, predictions=predict))


# In[ ]:


# plot_confusion_matrix(cm, [3, 4, 5, 6, 7])


# In[ ]:


# (img_train, label_train), (img_test, label_test) = tf.keras.datasets.fashion_mnist.load_data()


# In[ ]:


# img_train.shape


# In[ ]:


# label_train.shape


# In[ ]:




