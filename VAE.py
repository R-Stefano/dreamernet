import tensorflow.contrib.layers as nn
import tensorflow as tf
import numpy as np

class VAE():
    def __init__(self,sess, isTraining):
        self.sess=sess
        self.X=tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        self.latentDimension = 32
        self.model_folder='models/VAE/'


        self.buildGraph()
        self.buildLoss()
        self.buildUtils()
        self.sess.run(tf.global_variables_initializer())

        #Save/restore only the weights variables
        vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.saver=tf.train.Saver(var_list=vars)

        if not(isTraining):
            self.saver.restore(self.sess, self.model_folder+"graph.ckpt")
            print('VAE weights have been restored')

    
    def buildGraph(self):
        self.norm_x=self.X / 255.
        #Encoder
        with tf.variable_scope('encoder'):
            enc_1=nn.conv2d(self.norm_x, 32, 4, stride=2, padding="VALID")
            enc_2=nn.conv2d(enc_1, 64, 4, stride=2, padding="VALID")
            enc_3=nn.conv2d(enc_2, 128, 4, stride=2, padding="VALID")
            enc_4=nn.conv2d(enc_3, 256, 4, stride=2, padding="VALID")

            enc_4_flat=nn.flatten(enc_4)
        
        #Latent space
        with tf.variable_scope('latent_space'):
            self.mean = nn.fully_connected(enc_4_flat, self.latentDimension, activation_fn=None)
            #Leave RELu to keep std always positive. Or use tf.exp(tf.log(self.std))
            self.std = nn.fully_connected(enc_4_flat, self.latentDimension)

            self.latent = self.mean + self.std * tf.random.normal([self.latentDimension])

        #Decoder
        with tf.variable_scope('decoder'):
            dec_1_flat=nn.fully_connected(self.latent, enc_4_flat.get_shape().as_list()[1])
            dec_1_exp=tf.expand_dims(tf.expand_dims(dec_1_flat, -2), -2)
            
            dec_2=nn.conv2d_transpose(dec_1_exp, 128, 5, 2, padding="VALID")
            dec_3=nn.conv2d_transpose(dec_2, 64, 5, 2, padding="VALID")
            dec_4=nn.conv2d_transpose(dec_3, 32, 6, 2, padding="VALID")
            self.output=nn.conv2d_transpose(dec_4, 3, 6, 2, padding="VALID", activation_fn=tf.nn.sigmoid)
    
    def buildLoss(self):
        with tf.variable_scope('reconstruction_loss'):
            self.reconstr_loss=tf.reduce_mean(tf.reduce_sum(tf.square(self.norm_x - self.output), axis=[1,2,3]))
        with tf.variable_scope('KL_loss'):
            self.KLLoss= tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + tf.log(self.std +1e-9) - tf.square(self.mean) - self.std ,axis=-1))

        self.totLoss = self.reconstr_loss + self.KLLoss

        self.opt=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.totLoss)
    
    def buildUtils(self):
        #Create file
        self.file=tf.summary.FileWriter(self.model_folder, self.sess.graph)

        self.training=tf.summary.merge([
            tf.summary.scalar('reconstruction_loss', self.reconstr_loss),
            tf.summary.scalar('KL_loss', self.KLLoss),
            tf.summary.scalar('Total_loss', self.totLoss)
        ])

        self.recLossPlace=tf.placeholder(tf.float32)
        self.klLossPlace=tf.placeholder(tf.float32)
        self.totLossPlace=tf.placeholder(tf.float32)

        self.testTensorboard=tf.summary.merge([
            tf.summary.scalar('test_reconstruction_loss',self.recLossPlace),
            tf.summary.scalar('test_KL_loss',self.klLossPlace),
            tf.summary.scalar('test_Total_loss',self.totLossPlace)
        ])
    
    def predictBatch(self, sess, states):
        representations=[]
        for batchStart in range(0, len(states), 32):
            batchEnd=batchStart+32

            out=sess.run(self.latent, feed_dict={self.X: states[batchStart:batchEnd]})
            representations.extend(out.tolist())
        return representations

    def predict(self, state):
        out=self.sess.run(self.latent, feed_dict={self.X: np.expand_dims(state, axis=0)})

        return out
    
    def embedDecod(self, embed):
        out=self.sess.run(self.output, feed_dict={self.latent: embed})

        return out
