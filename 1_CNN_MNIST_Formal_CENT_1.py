## The script implements a CNN network. The CNN is trained and tested for classifying MNIST handwritten digits. 

## The network should firstly be trained (by setting "is_train" to True). After training, the model is automatically saved. The user can then test the model by setting "is_train" to False and "is_test" to True. And a testing accuracy of around 99% should be observed. Finally, by setting "cal_CENT" to True, the outputs of convolution layers will be saved to a folder. The user can use other scripts to compute the CENT.

## CNN Prototype: https://github.com/aymericdamien/TensorFlow-Examples/, Author:Aymeric Damien. The implemented CNN in this course project is a little different from the prototype mentioned above.

from __future__ import division, print_function, absolute_import

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
from scipy.stats import entropy
import os
import tensorflow as tf

# Training Parameters
learning_rate = 0.001
batch_size = 128
batch_size_test=10000
n_repeat = int(60000/batch_size)
epoch_pre = 2000
epoch_test = 20

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit


# Create the neural network
class Conv_Net():
    # Define a scope for reusing the variables
    def __init__(self, reuse, is_training, n_classes, dropout, learning_rate, batch_size):
        self.reuse=reuse
        self.is_training=is_training
        self.n_classes=n_classes
        self.dropout=dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        #name assumption:    name_scope/var_scope_name/var_name:0
        #example:            Train/ConvNet/Placeholder:0
        with tf.variable_scope('ConvNet', reuse=self.reuse):
            # TF Estimator input is a dict, in case of multiple inputs

            # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
            # Reshape to match picture format [Height x Width x Channel]
            # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
            self.X_ = tf.reshape(self.X, shape=[-1, 28, 28, 1])

            # Convolution Layer with 10 filters and a kernel size of 5
            self.conv1_ = tf.layers.conv2d(self.X_, 10, 5, activation=None, name='Conv1')
            self.conv1 = tf.nn.relu(self.conv1_)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            self.conv1 = tf.layers.max_pooling2d(self.conv1, 2, 2)

            print('self.conv1_:', self.conv1_)

            # Convolution Layer with 10 filters and a kernel size of 3
            self.conv2_ = tf.layers.conv2d(self.conv1, 10, 3, activation=None, name='Conv2')
            self.conv2 = tf.nn.relu(self.conv2_)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            self.conv2 = tf.layers.max_pooling2d(self.conv2, 2, 2)

            # Flatten the data to a 1-D vector for the fully connected layer
            self.fc1 = tf.contrib.layers.flatten(self.conv2)

            # Fully connected layer (in tf contrib folder for now)
            self.fc1 = tf.layers.dense(self.fc1, 1024)


            # Apply Dropout (if is_training is False, dropout is not applied)
            self.fc1 = tf.layers.dropout(self.fc1, rate=self.dropout, training=self.is_training)

            # Output layer, class prediction
            self.out = tf.layers.dense(self.fc1, self.n_classes)

            self.pred_classes = tf.argmax(self.out, axis=1)
            self.pred_probas = tf.nn.softmax(self.out)

            self.y = tf.placeholder(dtype=tf.int32, shape=[None])
            print('self.out:', self.out)
            print('self.y:', self.y)
            self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.out, labels=self.y))
            print('self.loss:', self.loss_op)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss_op,global_step=tf.train.get_global_step())


X_train, y_train = mnist.train.next_batch(60000)  #X shape: 60000*784,  y shape: 60000*1
X_test, y_test = mnist.test.next_batch(10000)    # X shape: 10000*784,  y shape: 10000*1

def main(is_train, is_test, save_path_, cal_CENT):
    # start the experiment from training phase
    g = tf.Graph()
    with g.as_default():
        with tf.name_scope("Train"):
            model_train = Conv_Net(reuse=False, is_training=True, n_classes=10, dropout=dropout,
                                   learning_rate=learning_rate, batch_size=batch_size)

        with tf.name_scope("Test"):
            model_test = Conv_Net(reuse=True, is_training=False, n_classes=10, dropout=dropout,
                                  learning_rate=learning_rate, batch_size=batch_size_test)
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

    if is_train:
        with tf.Session(graph=g) as sess:
            sess.run(init)
            print('=======clean network training:=================')
            for i in np.arange(epoch_pre):
                X_train_batch = X_train[(i%n_repeat)*batch_size:(i%n_repeat+1)*batch_size,:]
                y_train_batch = y_train[(i%n_repeat)*batch_size:(i%n_repeat+1)*batch_size]
                _, loss, preds_train = sess.run([model_train.train_op, model_train.loss_op, model_train.pred_classes],feed_dict={model_train.X:X_train_batch, model_train.y:y_train_batch})
                if (i%100)==0:
                    print('clean train epoch:', i, '     loss:',loss, '     accuracy:', np.sum(np.array(preds_train)==np.array(y_train_batch))/1.0/batch_size)
            if not os.path.exists(save_path_):
                os.makedirs(save_path_)
            save_path = saver.save(sess, save_path_, global_step=epoch_pre)

            #sess.close()
    # skip the training phase and start the experiment from the testing phase
    if is_test:
        with tf.Session(graph=g) as sess:
            saver.restore(sess, tf.train.latest_checkpoint('trained_model'))
            print('===========clean network testing===============')
            total_loss_clean = 0

            loss_test, preds_test = sess.run([model_test.loss_op, model_test.pred_classes],
                                             feed_dict={model_test.X: X_test, model_test.y: y_test})

            total_loss_clean += loss_test
            print('clean net total test loss:', total_loss_clean)
            print('testing accuracy:', np.sum(np.array(preds_test) == np.array(y_test)) / 1.0 / batch_size_test)

    if cal_CENT:

        with tf.Session(graph=g) as sess:
            saver.restore(sess, tf.train.latest_checkpoint('trained_model'))
            train_conv_1, train_conv_2 = sess.run([model_test.conv1_, model_test.conv2_],feed_dict={model_test.X: X_train, model_test.y: y_train})
            test_conv_1, test_conv_2 = sess.run([model_test.conv1_, model_test.conv2_], feed_dict={model_test.X: X_test, model_test.y: y_test})

        save_output_path = 'filter_output'
        if not os.path.exists(save_output_path):
            os.makedirs(save_output_path)
        np.save(save_output_path + '/X_train',X_train)
        np.save(save_output_path + '/y_train', y_train)
        np.save(save_output_path + '/X_test', X_test)
        np.save(save_output_path + '/y_test', y_test)
        np.save(save_output_path + '/train_conv_1', train_conv_1)
        np.save(save_output_path + '/train_conv_2', train_conv_2)
        np.save(save_output_path + '/test_conv_1', test_conv_1)
        np.save(save_output_path + '/test_conv_2', test_conv_2)

is_train=False   # start training the CNN by setting this to True. If the script is run for the first time, set this to True.
is_test=False    # test whether the network is trained successfully by setting this to True. This need to load from a trained model. Besure to train the network before testing it
cal_CENT=True    # whether to save the layer outputs.
save_path = "trained_model/CNN_1.ckpt"

main(is_train=is_train,is_test=is_test,  save_path_=save_path, cal_CENT=cal_CENT)






