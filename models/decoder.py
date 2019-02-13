import tensorflow.contrib.layers as nn
import tensorflow as tf
import numpy as np
class Decoder():
    def __init__(self,sess):
        self.sess=sess
        self.latentDimension = 32
        self.model_folder='models/VAE/'
        self.X=tf.placeholder(tf.float32, shape=[None, self.latentDimension])


        self.buildGraph()
        self.sess.run(tf.global_variables_initializer())

        #Save/restore only the weights variables
        vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
        print(vars)
        for var in vars:
            print(var)
        self.saver=tf.train.Saver(var_list=vars)

        self.saver.restore(self.sess, self.model_folder+"graph.ckpt")
        print('Predictor weights have been restored')

    
    def buildGraph(self):
        #Decoder
        with tf.variable_scope('decoder_predict'):
            dec_1_flat=nn.fully_connected(self.X, 1024)
            dec_1_exp=tf.expand_dims(tf.expand_dims(dec_1_flat, -2), -2)
            
            dec_2=nn.conv2d_transpose(dec_1_exp, 128, 5, 2, padding="VALID")
            dec_3=nn.conv2d_transpose(dec_2, 64, 5, 2, padding="VALID")
            dec_4=nn.conv2d_transpose(dec_3, 32, 6, 2, padding="VALID")
            self.output=nn.conv2d_transpose(dec_4, 3, 6, 2, padding="VALID", activation_fn=tf.nn.sigmoid)

    def predict(self, state):
        out=self.sess.run(self.output, feed_dict={self.X: np.expand_dims(state, axis=0)})

        return out