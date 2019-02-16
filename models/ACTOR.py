import tensorflow.contrib.layers as nn
import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
class ACTOR():
    def __init__(self,sess):
        self.sess=sess
        self.model_folder='models/ACTOR/'

        self.latent_dimension=FLAGS.latent_dimension
        self.num_actions=FLAGS.num_actions

        self.X=tf.placeholder(tf.float32, shape=[None, self.latent_dimension])
        self.buildGraph()
        self.buildLoss()
        self.sess.run(tf.global_variables_initializer())

        #Save/restore only the weights variables
        vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.saver=tf.train.Saver(var_list=vars)

        if not(FLAGS.training_ACTOR):
            self.saver.restore(self.sess, self.model_folder+"graph.ckpt")
            print('ACTOR weights have been restored')
        else:
            self.buildUtils()

    
    def buildGraph(self):
        x=self.X
        self.setups={
            'MLP': [nn.fully_connected(x, 128), nn.fully_connected(x, 256)]
        }

        for l in self.setups['MLP']:
            x=l

        self.policyOutput=nn.fully_connected(x, self.num_actions, activation_fn=tf.nn.softmax)
        self.valueOutput=nn.fully_connected(x, 1, activation_fn=None)
    
    def buildLoss(self):
        self.actions=tf.placeholder(tf.int32)
        self.Vs1=tf.placeholder(tf.float32)
        self.rewards=tf.placeholder(tf.float32)
        self.isTerminal=tf.placeholder(tf.float32)

        #Convert actions to hot encode
        self.a_hot_encoded=tf.one_hot(self.actions, FLAGS.num_actions)
        
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
            tf.summary.scalar('Actor_policy_loss', self.policyLoss),
            tf.summary.scalar('Actor_value_loss', self.valueLoss),
            tf.summary.scalar('Actor_tot_loss', self.totLoss)
        ])

        self.avgRew=tf.placeholder(tf.float32)
        self.testing=tf.summary.merge([
            tf.summary.scalar('Actor_avg_reward', self.avgRew)
        ])
    def save(self):
        self.saver.save(self.sess, self.model_folder+"graph.ckpt")

    def predict(self, state):
        if (len(state.shape) == 1):
            state=np.expand_dims(state, axis=0)

        policy, value=self.sess.run([self.policyOutput, self.valueOutput], feed_dict={self.X: state})

        return np.squeeze(policy), np.squeeze(value)
