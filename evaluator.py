import tensorflow.contrib.layers as nn
import tensorflow as tf
import numpy as np

class Evaluator():
    def __init__(self,sess, isTraining):
        self.sess=sess
        self.latentDimension = 32
        self.num_actions=3
        self.X=tf.placeholder(tf.float32, shape=[None, self.latentDimension])
        self.model_folder='models/predictor/'


        self.buildGraph()
        self.buildLoss()
        self.buildUtils()
        self.sess.run(tf.global_variables_initializer())

        #Save/restore only the weights variables
        vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.saver=tf.train.Saver(var_list=vars)

        if not(isTraining):
            self.saver.restore(self.sess, self.model_folder+"graph.ckpt")
            print('Predictor weights have been restored')

    
    def buildGraph(self):
        l1 = nn.fully_connected(self.X, 128)
        l2 = nn.fully_connected(l1, 256)

        self.policyOutput=nn.fully_connected(l2, self.num_actions, activation_fn=tf.nn.softmax)
        self.valueOutput=nn.fully_connected(l2, 1, activation_fn=None)
    
    def buildLoss(self):
        self.actions=tf.placeholder(tf.int32)
        self.Vs1=tf.placeholder(tf.float32)
        self.rewards=tf.placeholder(tf.float32)
        self.isTerminal=tf.placeholder(tf.float32)
        #Convert actions to hot encode
        self.a_hot_encoded=tf.one_hot(self.actions, self.num_actions)
        
        with tf.variable_scope('policy_loss'):
            self.policyLoss=tf.reduce_mean(-tf.reduce_sum(self.a_hot_encoded * tf.log(self.policyOutput + 1e-9) + (1-self.a_hot_encoded)*tf.log(1-self.policyOutput + 1e-9),axis=-1))
        
        with tf.variable_scope('value_loss'):
            self.valueLoss=tf.reduce_mean(tf.square(self.valueOutput - (1-self.isTerminal)*(self.rewards + 0.99 * self.Vs1)))

        self.totLoss = self.valueLoss + self.policyLoss

        self.opt=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.totLoss)
        
    
    def buildUtils(self):
        #Create file
        self.file=tf.summary.FileWriter(self.model_folder, self.sess.graph)
        
        self.training=tf.summary.merge([
            tf.summary.scalar('policy_loss', self.policyLoss),
            tf.summary.scalar('value_loss', self.valueLoss),
            tf.summary.scalar('Total_loss', self.totLoss)
        ])

    def predict(self, state):
        if (len(state.shape) == 1):
            state=np.expand_dims(state, axis=0)

        policy, value=self.sess.run([self.policyOutput, self.valueOutput], feed_dict={self.X: state})

        return policy, value
