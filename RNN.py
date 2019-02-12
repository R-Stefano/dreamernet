import tensorflow.contrib.layers as nn
import tensorflow as tf
class RNN():
    def __init__(self, sess, isTraining):
        self.sess=sess
        self.state_rep_length=32 #Same as VAE LATENT VEC
        self.timesteps=10 #how many frames in a sequence
        self.hidden_units=128
        self.model_folder='models/RNN/'
        self.X=tf.placeholder(tf.float32, shape=[None, self.timesteps, self.state_rep_length+1])
        
        self.buildGraph()
        self.buildLoss()
        self.buildUtils()
        self.sess.run(tf.global_variables_initializer())

        #Save/restore only the weights variables
        vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.saver=tf.train.Saver(var_list=vars)

        if not(isTraining):
            self.saver.restore(self.sess, self.model_folder+"graph.ckpt")
            print('RNN weights have been restored')

    
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

        self.next_state=nn.fully_connected(self.output[:,-1], self.state_rep_length, activation_fn=None)

    
    def buildLoss(self):
        self.true_next_state=tf.placeholder(tf.float32, shape=[None, self.state_rep_length])

        #print MSE
        self.loss=tf.reduce_mean(tf.reduce_sum(tf.square(self.true_next_state - self.next_state), axis=-1))

        self.opt=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)
    
    def buildUtils(self):
        self.file=tf.summary.FileWriter(self.model_folder, self.sess.graph)
        self.training=tf.summary.merge([
            tf.summary.scalar('Loss', self.loss)
        ])

        self.totLossPlace=tf.placeholder(tf.float32)

        self.testing=tf.summary.merge([
            tf.summary.scalar('test_loss',self.totLossPlace)
        ])
    
    def predict(self, input, cell, hidden):
        nextStates=self.sess.run(self.next_state, feed_dict={self.X: input,
                                                             self.cell_state: cell,
                                                             self.hidden_state: hidden})
        return nextStates