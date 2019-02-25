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
        self.train_size=FLAGS.RNN_train_size

        self.sequence_length=FLAGS.sequence_length
        self.latent_dimension=FLAGS.latent_dimension
        self.hidden_units=FLAGS.hidden_units
        self.num_layers=FLAGS.LSTM_layers
        self.num_components=FLAGS.num_components
        self.prediction=FLAGS.prediction_type

        self.X=tf.placeholder(tf.float32, shape=[None, None, self.latent_dimension+ FLAGS.actions_size])
        
        self.buildGraph()
        self.buildLoss()
        self.buildUtils()

        self.saver=tf.train.Saver()

        if (not(FLAGS.training_RNN) or not(FLAGS.preprocessing)):
            self.saver.restore(self.sess, self.model_folder+"graph.ckpt")
            print('RNN weights have been restored')
        else:
            self.sess.run(tf.global_variables_initializer())

    
    def buildGraph(self):
        #initialize init state
        self.init_state=tf.placeholder(tf.float32, [self.num_layers, 2, None, self.hidden_units])

        with tf.variable_scope('LSTM'):        
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
            #hidden_cell_statesTuple is a list of 2 elements: self.hidden_state, self.cell_state where self.output[:, -1, :]=self.cell_state
            self.output, self.hidden_cell_statesTuple=tf.nn.dynamic_rnn(cell=self.lstm_cell, inputs=self.X, initial_state=init_state)

        flat=tf.reshape(self.output, (-1, self.hidden_units), name="flat_LSTM_output")

        with tf.variable_scope('MLP'):        
            self.flat1=nn.fully_connected(flat, 256)

        with tf.variable_scope('state_output'): 
            if (self.prediction=='MSE'):
                self.mean=nn.fully_connected(self.flat1, self.latent_dimension)
                self.stddev=nn.fully_connected(self.flat1, self.latent_dimension, activation_fn=tf.nn.softplus)
                self.next_state_out=self.mean + self.stddev * tf.random.normal([self.latent_dimension])
            elif (self.prediction=='GMM'):    
                self.mean=nn.fully_connected(self.flat1, self.num_components*self.latent_dimension)
                self.stddev=nn.fully_connected(self.flat1, self.num_components*self.latent_dimension)#, activation_fn=tf.nn.softplus)
                self.logmix=nn.fully_connected(self.flat1, self.num_components*self.latent_dimension)
            elif (self.prediction=='KL'):
                self.mean=nn.fully_connected(self.flat1, self.latent_dimension)
                self.stddev=nn.fully_connected(self.flat1, self.latent_dimension, activation_fn=tf.nn.softplus)    
                self.next_state_out=self.mean + self.stddev * tf.random.normal([self.latent_dimension])
        
        with tf.variable_scope('reward_output'):
            self.reward_out=nn.fully_connected(self.flat1, 1)
    
    def buildLoss(self):
        if (self.prediction=='MSE' or self.prediction=='GMM'):
            self.true_next_state=tf.placeholder(tf.float32, shape=[None, None, self.latent_dimension+1])
        elif (self.prediction=='KL'):
            self.true_next_state=tf.placeholder(tf.float32, shape=[None, None, 2*self.latent_dimension+1])

        with tf.variable_scope('prepare_labels'):
            if (self.prediction=='MSE' or self.prediction=='GMM'):
                true_next=tf.reshape(self.true_next_state, (-1, self.latent_dimension+1))
                true_next_state, true_reward=tf.split(true_next, [self.latent_dimension,1], 1)
                if (self.prediction=='GMM'):
                    true_next_state=tf.reshape(true_next_state, [-1,1])
            elif (self.prediction=='KL'):
                true_next=tf.reshape(self.true_next_state, (-1, 2*self.latent_dimension+1))
                self.true_next_state_mu, self.true_next_state_std, true_reward=tf.split(true_next, [self.latent_dimension,self.latent_dimension,1], 1)

        with tf.variable_scope('representation_loss'):
            if (self.prediction=='MSE'):
                self.representation_loss=tf.reduce_mean(tf.reduce_sum(tf.square(true_next_state - self.next_state_out),axis=-1))
            elif (self.prediction=='GMM'): 
                self.mean=tf.reshape(self.mean , [-1, self.num_components])
                self.stddev=tf.reshape(self.stddev , [-1, self.num_components])
                self.logmix=tf.reshape(self.logmix , [-1, self.num_components])
                logmix=self.logmix - tf.reduce_logsumexp(self.logmix, 1, keepdims=True)
                #tf_lognormal
                self.lognorm=-0.5 * ((true_next_state - self.mean) / tf.exp(self.stddev)) ** 2 - self.stddev - np.log(np.sqrt(2.0 * np.pi))
                v= self.logmix + self.lognorm
                self.representation_loss= -tf.reduce_mean(tf.reduce_logsumexp(v, 1, keepdims=True))
            elif (self.prediction=='KL'):
                self.single_loss=tf.reduce_sum(tf.log(self.stddev / (self.true_next_state_std + 1e-9)) +  ((self.true_next_state_std**2 + (self.mean - self.true_next_state_mu)**2)/(2*self.stddev**2)) -0.5, axis=-1)
                print('kl loss for single example CHECK IF IT IS POSITIVE', self.single_loss)
                self.representation_loss=tf.reduce_mean(self.single_loss)

        with tf.variable_scope('reward_loss'):        
            self.reward_loss=tf.reduce_mean(tf.square(true_reward -self.reward_out))

        self.loss=self.representation_loss + self.reward_loss

        self.opt=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)
    
    def buildUtils(self):
        self.file=tf.summary.FileWriter(self.model_folder, self.sess.graph)

        with tf.name_scope('RNN_train'):
            self.training=tf.summary.merge([
                tf.summary.scalar('state_loss', self.representation_loss),
                tf.summary.scalar('rew_loss', self.reward_loss),
                tf.summary.scalar('tot_loss', self.loss)
            ])

        with tf.name_scope('RNN_test'):
            self.testing=tf.summary.merge([
                tf.summary.scalar('state_loss', self.representation_loss),
                tf.summary.scalar('rew_loss', self.reward_loss),
                tf.summary.scalar('tot_loss', self.loss)
            ])

            self.frame_s=tf.placeholder(tf.float32)
            self.frame_s1=tf.placeholder(tf.float32)

            self.predicting=tf.summary.merge([
                tf.summary.image('frame_s', self.frame_s),
                tf.summary.image('frame_s1', self.frame_s1)
            ])
        
        with tf.name_scope('RNN_playing'):
            self.playing=tf.summary.merge([
                tf.summary.scalar('state_loss', self.representation_loss),
                tf.summary.scalar('rew_loss', self.reward_loss),
                tf.summary.scalar('tot_loss', self.loss)
            ])
    
    def save(self):
        self.saver.save(self.sess, self.model_folder+"graph.ckpt")

    def predict(self, input, initialize=[]):
        if (len(initialize)==0):
            #initialize hidden state and cell state to zeros 
            initialize=np.zeros((self.num_layers, 2, input.shape[0], self.hidden_units))

        nextStates, rew, h_c_states=self.sess.run([self.next_state_out, 
                                                    self.reward_out,
                                                    self.hidden_cell_statesTuple], 
                                                    feed_dict={self.X: input,
                                                                self.init_state: initialize})

        return np.concatenate((nextStates,rew),axis=-1), np.expand_dims(np.asarray(h_c_states[0]), axis=0)