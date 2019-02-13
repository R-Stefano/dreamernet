import tensorflow.contrib.layers as nn
import tensorflow as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS
class RNN():
    def __init__(self, sess):
        self.sess=sess
        self.model_folder='models/RNN/'

        #HYPERPARAMETERS
        self.sequence_length=FLAGS.sequence_length
        self.latent_dimension=FLAGS.latent_dimension
        self.hidden_units=FLAGS.hidden_units

        self.X=tf.placeholder(tf.float32, shape=[None, self.sequence_length, self.latent_dimension+1])
        
        self.buildGraph()
        self.buildLoss()
        self.sess.run(tf.global_variables_initializer())

        #Save/restore only the weights variables
        vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.saver=tf.train.Saver(var_list=vars)

        if not(FLAGS.training_RNN):
            self.saver.restore(self.sess, self.model_folder+"graph.ckpt")
            print('RNN weights have been restored')
        else:
            self.buildUtils()

    
    def buildGraph(self):
        #initialize init state
        self.cell_state = tf.placeholder(tf.float32)
        self.hidden_state = tf.placeholder(tf.float32)
        init_state = tf.nn.rnn_cell.LSTMStateTuple(self.cell_state, self.hidden_state)
        
        #Define the single LSTM cell with the number of hidden units
        self.lstm_cell=tf.contrib.rnn.LSTMCell(self.hidden_units, name="LSTM_Cell")

        #Feed an input of shape [batch_size, 10, state_rep_length]
        #self.output is a tensor of shape [batch_size, 100, hidden_units]
        #hidden_cell_statesTuple is a list of 2 elements: self.cell_state, self.hidden_state where self.output[:, -1, :]=self.cell_state
        self.output, self.hidden_cell_statesTuple=tf.nn.dynamic_rnn(cell=self.lstm_cell, inputs=self.X, initial_state=init_state)

        self.next_state=nn.fully_connected(self.output[:,-1], self.latent_dimension, activation_fn=None)

    
    def buildLoss(self):
        self.true_next_state=tf.placeholder(tf.float32, shape=[None, self.latent_dimension])

        #print MSE
        self.loss=tf.reduce_mean(tf.reduce_sum(tf.square(self.true_next_state - self.next_state), axis=-1))

        self.opt=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)
    
    def buildUtils(self):
        self.file=tf.summary.FileWriter(self.model_folder, self.sess.graph)
        self.training=tf.summary.merge([
            tf.summary.scalar('RNN_loss', self.loss)
        ])

        self.totLossPlace=tf.placeholder(tf.float32)

        self.testing=tf.summary.merge([
            tf.summary.scalar('RNN_test_loss',self.totLossPlace)
        ])
    
    def save(self):
        self.saver.save(self.sess, self.model_folder+"graph.ckpt")

    def predict(self, input, initialize=None):
        if (initialize==None):
            #initialize hidden state and cell state to zeros 
            cell_s=np.zeros((input.shape[0], self.hidden_units))
            hidden_s=np.zeros((input.shape[0], self.hidden_units))

        nextStates=self.sess.run(self.next_state, feed_dict={self.X: input,
                                                             self.cell_state: cell_s,
                                                             self.hidden_state: hidden_s})
        return nextStates