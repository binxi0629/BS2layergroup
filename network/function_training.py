import os
import datetime
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras import Model

from matplotlib import pyplot as plt


def train_it(device, model, save_path,
             epochs, train_loader, test_loader, loss_fn, optimizer,
             train_loss, train_accuracy, test_loss, test_accuracy, eval_num_epochs=5, numClasss=20,summary=True):

    @tf.function
    def train_step(x, y_true):  # x shape: (32, 100, 400) labels:shape (32, 1)

        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            tmp_onehot = tf.one_hot(y_true,depth=numClasss)
            y_true_onehot = tf.reshape(tmp_onehot, shape=(tmp_onehot.get_shape()[0],numClasss))

            loss = loss_fn(y_true_onehot, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(y_true, predictions)

    @tf.function
    def test_step(x, y_true):
        predictions = model(x, training=False)

        tmp_onehot = tf.one_hot(y_true, depth=numClasss)
        y_true_onehot = tf.reshape(tmp_onehot, shape=(tmp_onehot.get_shape()[0], numClasss))

        t_loss = loss_fn(y_true_onehot, predictions)

        test_loss(t_loss)
        test_accuracy(y_true, predictions)

    losses_train = []
    accuracy_train = []
    losses_test =[]
    accuracy_test = []
    # summary
    print(model.summary())
    if summary:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        train_log_dir = 'logs/gradient_tape/'+current_time+'/train'
        test_log_dir = 'logs/gradient_tape/'+current_time+'/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # train
    for epoch in range(epochs):

        for x, labels in train_loader:
            # print("x shape:", x.shape)
            # print("labels:shape",labels.shape)
            train_step(x, labels)

        losses_train.append(train_loss.result())
        accuracy_train.append(train_accuracy.result())
        print(f'Epoch {epoch + 1} | Loss:   {train_loss.result()} | Accuracy (%):   {train_accuracy.result() * 100}')

        if summary:
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy',train_accuracy.result(), step=epoch)

        if (epoch+1) % eval_num_epochs == 0:

            for test_x, test_labels in test_loader:
                test_step(test_x, test_labels)

                if summary:
                    with test_summary_writer.as_default():
                        tf.summary.scalar('loss',test_loss.result(), step=epoch)
                        tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

                losses_test.append(test_loss.result())
                accuracy_test.append(test_accuracy.result())

            print(f'\tTest Loss:  {test_loss.result()}  |  Test Accuracy (%): {test_accuracy.result()*100}')

        # Reset the metrics at the start of next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

    # save model
    # tf.saved_model.save(model, save_path)
    tf.keras.models.save_model(model=model, filepath=save_path)

    return losses_train, accuracy_train, losses_test, accuracy_test, model

def confusion_matrix(labels, predictions, class_names):
    cm = np.array(tf.math.confusion_matrix(labels=labels, predictions=predictions))

    def plot_confusion_matrix(cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
           cm (array, shape = [n, n]): a confusion matrix of integer classes
           class_names (array, shape = [n]): String names of the integer classes
        """
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

        figure = plt.figure(figsize=(8, 8), dpi=400)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
    #     plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
    #     plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)


        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > threshold else "black"
                plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Prediction')
    #     return figure
    
    plot_confusion_matrix(cm, class_names)