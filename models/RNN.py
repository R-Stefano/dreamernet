import tensorflow as tf
import tensorflow.contrib.layers as nn

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
        self.num_layers=FLAGS.LSTM_layers

        self.X=tf.placeholder(tf.float32, shape=[None, None, self.latent_dimension+1])
        
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
        self.init_state=tf.placeholder(tf.float32, [self.num_layers, 2, None, self.hidden_units])

        init_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
            self.init_state[l][0],
            self.init_state[l][1]) 
            for l in range(self.num_layers)])

        layers=[]
        for i in range(self.num_layers):
            #Define the LSTM cell with the number of hidden units
            cell=tf.nn.rnn_cell.LSTMCell(self.hidden_units, name="LSTM_"+str(i))
            layers.append(cell)

        self.lstm_cell = tf.contrib.rnn.MultiRNNCell(layers)

        #Feed an input of shape [batch_size, 10, state_rep_length]
        #self.output is a tensor of shape [batch_size, 100, hidden_units]
        #hidden_cell_statesTuple is a list of 2 elements: self.cell_state, self.hidden_state where self.output[:, -1, :]=self.cell_state
        self.output, self.hidden_cell_statesTuple=tf.nn.dynamic_rnn(cell=self.lstm_cell, inputs=self.X, initial_state=init_state)

        flat=tf.reshape(self.output, (-1, self.hidden_units))
        self.flat1=nn.fully_connected(flat, 256)

        self.mean=nn.fully_connected(self.flat1, self.latent_dimension + 1)
        self.stddev=nn.fully_connected(self.flat1, self.latent_dimension + 1, activation_fn=tf.nn.softplus)

        self.next_state=self.mean + self.stddev * tf.random.normal([self.latent_dimension +1])
    
    def buildLoss(self):
        self.true_next_state=tf.placeholder(tf.float32, shape=[None, None, self.latent_dimension+1])
        true_next=tf.reshape(self.true_next_state, (-1, self.latent_dimension+1))

        #print MSE
        self.loss=tf.reduce_mean(tf.reduce_sum(tf.square(true_next - self.next_state), axis=-1))

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
            init_state=np.zeros((self.num_layers, 2, input.shape[0], self.hidden_units))

        nextStates, hidden_state=self.sess.run([self.next_state,self.hidden_cell_statesTuple], feed_dict={self.X: input,
                                                             self.init_state: init_state})
        return nextStates, hidden_state[-1]