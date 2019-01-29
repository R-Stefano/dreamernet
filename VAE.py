import tensorflow.contrib.layers as nn
import tensorflow as tf
class VAE():
    def __init__(self):
        self.X=tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        self.latentDimension = 32
        self.buildGraph()
        self.buildLoss()
        self.buildUtils()
    
    def buildGraph(self):
        #Encoder
        with tf.variable_scope('encoder'):
            enc_1=nn.conv2d(self.X, 32, 4, stride=2, padding="VALID")
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
            self.reconstr_loss=tf.reduce_mean(tf.reduce_sum(tf.square(self.X - self.output), axis=[1,2,3]))
        with tf.variable_scope('KL_loss'):
            self.KLLoss= tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + tf.log(self.std +1e-9) - tf.square(self.mean) - self.std ,axis=-1))

        self.totLoss = self.reconstr_loss + self.KLLoss

        self.opt=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.totLoss)
    
    def buildUtils(self):
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
