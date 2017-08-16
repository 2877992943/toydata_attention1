#encoding:utf-8

'''
A Dynamic Recurrent Neural Network (LSTM) implementation example using
TensorFlow library. This example is using a toy dataset to classify linear
sequences. The generated sequences have variable length.

Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import random
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn



from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import dynamic_rnn as rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

from attention import attention
from utils import *

VOCAB_SIZE=2000
EMBEDDING_SIZE=64
ATTENTION_SIZE=50
HIDDEN_SIZE=64
KEEP_PROB = 0.8

# ====================
#  TOY DATA GENERATOR
# ====================
class ToySequenceData(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])

    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, n_samples=10000, max_seq_len=20, min_seq_len=3,
                 max_value=1000):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            # Random sequence length
            len = random.randint(min_seq_len, max_seq_len)
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(len)
            # Add a random or linear int sequence (50% prob)
            if random.random() < .5:
                # Generate a linear sequence
                rand_start = random.randint(0, max_value - len)
                s = [int(i) for i in
                     range(rand_start, rand_start + len)]
                # Pad sequence for dimension consistency
                s += [0 for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([0.,1.])
            else:
                # Generate a random sequence
                s = [random.randint(0, max_value)
                     for i in range(len)]
                # Pad sequence for dimension consistency
                s += [0 for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([1.,0.])
        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen




def dynamicRNN(x_ph,y_ph, seqlen_ph): # x1 [batch,steps]

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    ####
    ## embed
    with tf.device("/cpu:0"):
        embedding = tf.get_variable(
            "embedding", [VOCAB_SIZE, EMBEDDING_SIZE],dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, x_ph)#inputs [batch,20steps,64hid]

    # if is_training and config.keep_prob < 1:
    #     inputs = tf.nn.dropout(inputs, config.keep_prob)

    # (Bi-)RNN layer(-s)
    rnn_outputs, _ = bi_rnn(GRUCell(HIDDEN_SIZE), GRUCell(HIDDEN_SIZE),
                            inputs=inputs, sequence_length=seqlen_ph, dtype=tf.float32)

    # Attention layer
    attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)#[batch,hid*2] [batch,20seqlen]

    # Dropout
    drop = tf.nn.dropout(attention_output, keep_prob_ph)

    # Fully connected layer
    W = tf.Variable(tf.truncated_normal([drop.get_shape()[1].value, n_classes], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[2]))
    y_hat = tf.nn.xw_plus_b(drop, W, b)
    #y_hat = tf.squeeze(y_hat)

    # Cross-entropy loss and optimizer initialization
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=y_ph))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

    # Accuracy metric
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(y_hat)), y_ph), tf.float32))

    return loss,optimizer,accuracy



    #inputs=tf.squeeze(inputs,axis=2)#[batch,20steps,64hid]
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # x = tf.unstack(inputs, seq_max_len, 1) #x dim[a,b,c,d]-> b [a,c,d]   20 ge [batch,64]
    #
    # # Define a lstm cell with tensorflow
    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    #
    #
    #
    # # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # # calculation.
    # outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x,dtype=tf.float32,
    #                             sequence_length=seqlen)# outputs 20 ge [batch,64]
    #
    # # When performing dynamic calculation, we must retrieve the last
    # # dynamically computed output, i.e., if a sequence length is 10, we need
    # # to retrieve the 10th output.
    # # However TensorFlow doesn't support advanced indexing yet, so we build
    # # a custom op that for each sample in batch size, get its length and
    # # get the corresponding relevant output.
    #
    # # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # # and change back dimension to [batch_size, n_step, n_input]
    # outputs1 = tf.stack(outputs)# [20steps,batch,64hid]
    # outputs2 = tf.transpose(outputs1, [1, 0, 2]) # [batch,steps20,hid64]
    # batch_size = tf.shape(outputs2)[0]
    # #outputs22=tf.reshape(outputs2,[batch_size,seq_max_len*n_hidden])#[batch,steps*hid]
    #
    # # Hack to build the indexing and retrieve the right output.
    #
    # # Start indices for each sample
    # index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1) # get last word output64  of each sample(steps_seqlen)
    # # Indexing
    # outputs3 = tf.gather(tf.reshape(outputs2, [-1, n_hidden]), index) #[batch,64]
    #
    # # Linear activation, using outputs computed above
    #
    # ret=tf.matmul(outputs3, weights['out']) + biases['out'] # [batch,64] [64,2] ->[batch,2]
    # return ret





# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.01
training_iters = 10000
batch_size = 128
display_step = 10

# Network Parameters
seq_max_len = 20 # Sequence max length
n_hidden = 64 # hidden layer num of features
n_classes = 2 # linear sequence or not


###

trainset = ToySequenceData(n_samples=10000, max_seq_len=seq_max_len)
testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)

# tf Graph input
x_ph = tf.placeholder("int32", [None, seq_max_len])
y_ph = tf.placeholder("float32", [None, n_classes])
# A placeholder for indicating each sequence length
seqlen_ph = tf.placeholder(tf.int32, [None])
keep_prob_ph = tf.placeholder(tf.float32)

cost, optimizer, accuracy= dynamicRNN(x_ph,y_ph, seqlen_ph)


# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x_ph: batch_x, y_ph: batch_y,
                                       seqlen_ph: batch_seqlen,keep_prob_ph:KEEP_PROB})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x_ph: batch_x, y_ph: batch_y,
                                                seqlen_ph: batch_seqlen,keep_prob_ph:KEEP_PROB})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x_ph: batch_x, y_ph: batch_y,
                                             seqlen_ph: batch_seqlen,keep_prob_ph:KEEP_PROB})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy
    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x_ph: test_data, y_ph: test_label,
                                      seqlen_ph: test_seqlen,keep_prob_ph:KEEP_PROB}))
