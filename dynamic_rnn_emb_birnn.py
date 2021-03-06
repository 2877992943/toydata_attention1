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
from tensorflow.contrib import rnn

VOCAB_SIZE=2000


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
                self.labels.append([0,1])
            else:
                # Generate a random sequence
                s = [random.randint(0, max_value)
                     for i in range(len)]
                # Pad sequence for dimension consistency
                s += [0 for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([1,0])
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




def dynamicRNN(x1, seqlen, weights, biases): # x1 [batch,steps]

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    ####
    ## embed
    with tf.device("/cpu:0"):
        embedding = tf.get_variable(
            "embedding", [VOCAB_SIZE, n_hidden],dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, x1)#[batch,20steps,64hid]

    # if is_training and config.keep_prob < 1:
    #     inputs = tf.nn.dropout(inputs, config.keep_prob)



    #inputs=tf.squeeze(inputs,axis=2)#[batch,20steps,64hid]
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(inputs, seq_max_len, 1) #x dim[a,b,c,d]-> b [a,c,d]   20 ge [batch,64]

    # Define a lstm cell with tensorflow
    #lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)



    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    #outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x,dtype=tf.float32,sequence_length=seqlen)# outputs 20 ge [batch,64]

    outputs, fw, bw = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                 dtype=tf.float32,sequence_length=seqlen)  # 28 ge [batch,64x2]

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs1 = tf.stack(outputs)# [20steps,batch,64hid]
    outputs2 = tf.transpose(outputs1, [1, 0, 2]) # [batch,steps20,hid64x2]
    batch_size = tf.shape(outputs2)[0]
    #outputs22=tf.reshape(outputs2,[batch_size,seq_max_len*n_hidden])#[batch,steps*hid]

    # Hack to build the indexing and retrieve the right output.

    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1) # get last word output64  of each sample(steps_seqlen)
    # Indexing
    outputs3 = tf.gather(tf.reshape(outputs2, [-1, n_hidden*2]), index) #[batch,64]

    # Linear activation, using outputs computed above

    ret=tf.matmul(outputs3, weights['out']) + biases['out'] # [batch,64] [64,2] ->[batch,2]
    return ret





# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.01
training_iters = 100000
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
x = tf.placeholder("int32", [None, seq_max_len])
y = tf.placeholder("int32", [None, n_classes])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden*2, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


######


pred = dynamicRNN(x, seqlen, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       seqlen: batch_seqlen})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,
                                                seqlen: batch_seqlen})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
                                             seqlen: batch_seqlen})
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
        sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                      seqlen: test_seqlen}))
