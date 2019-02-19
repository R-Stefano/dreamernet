import tensorflow.contrib.layers as nn
import tensorflow as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS
'''
WHAT CAN I TRIE TO DO
1. WARUP GENERATOR
train only the generator as if it was a VAE to create a good approximation and then train GAN

2. keep large filters in the first layers, 
like 5x5 in order to extract much inofrmatrion form the input image,
then decrease it to 3x3

3. real image=0 fake =1
it helps with the gradient. MOreover spoftlabels:
instead of 0 and 1, use {0-0.1} and  {0.9-1}
'''
class VAEGAN():
    def __init__(self,sess):
        self.sess=sess

        #HYPERPARAMETERS
        self.model_folder='models/VAEGAN/'
        self.img_size=FLAGS.img_size
        self.latent_dim=FLAGS.latent_dimension
        self.beta=FLAGS.beta
        self.gen_X=tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, 3])
        self.disc_Y=tf.placeholder(tf.float32, shape=[None])

        self.buildGraph()
        self.buildLoss()
        self.buildAccuracy()
        self.buildUtils()

        #Save/restore only the weights variables
        self.saver=tf.train.Saver()

        if not(FLAGS.training_VAE or FLAGS.training_GAN_Discriminator_real or FLAGS.training_GAN_Discriminator_fake):
            self.saver.restore(self.sess, self.model_folder+"graph.ckpt")
            print('VAE weights have been restored')
        else:
            self.sess.run(tf.global_variables_initializer())

    def buildGraph(self):
        with tf.variable_scope('generator'):
            #Encoder
            self.gen_norm_x=self.gen_X / 255.
            with tf.variable_scope('encoder'):
                enc_1=nn.conv2d(self.gen_norm_x, 32, 5, stride=2) #out :48
                enc_2=nn.conv2d(enc_1, 32, 5, stride=2) #out:24
                enc_3=nn.conv2d(enc_2, 64, 3, stride=2) #out: 12
                enc_4=nn.conv2d(enc_3, 128, 3, stride=2) #out: 6
                enc_5=nn.conv2d(enc_4, 256, 3, stride=3) #out: 2
                enc_4_flat=nn.flatten(enc_5)
            
            #Latent space
            with tf.variable_scope('latent_space'):
                self.mean = nn.fully_connected(enc_4_flat, self.latent_dim, activation_fn=None)
                #Leave RELu to keep std always positive. Or use tf.exp(tf.log(self.std))
                self.std = nn.fully_connected(enc_4_flat, self.latent_dim, activation_fn=tf.nn.softplus)

                self.latent = self.mean + self.std * tf.random.normal([self.latent_dim])

            #Decoder
            with tf.variable_scope('decoder'):
                dec_1_flat=nn.fully_connected(self.latent, enc_4_flat.get_shape().as_list()[1])
                dec_1_exp=tf.expand_dims(tf.expand_dims(dec_1_flat, -2), -2)
                dec_2=nn.conv2d_transpose(dec_1_exp, 256, 3, 2) #out: 2
                dec_3=nn.conv2d_transpose(dec_2, 128, 3, 3) #out: 6
                dec_4=nn.conv2d_transpose(dec_3, 64, 3, 2) #out: 12
                dec_5=nn.conv2d_transpose(dec_4, 32, 5, 2) #out: 24
                dec_6=nn.conv2d_transpose(dec_5, 32, 5, 2) #out: 48

                self.gen_output=nn.conv2d_transpose(dec_6, 3, 3, 2, activation_fn=tf.nn.sigmoid)

        with tf.variable_scope('Discriminator'): 
            enc_1=nn.conv2d(self.gen_output, 32, 5, stride=2) #out :48
            enc_2=nn.conv2d(enc_1, 32, 5, stride=2) #out:24
            enc_3=nn.conv2d(enc_2, 64, 5, stride=2) #out: 12
            enc_4=nn.conv2d(enc_3, 128, 3, stride=2) #out: 6
            enc_5=nn.conv2d(enc_4, 256, 3, stride=3) #out: 2 
            enc_5_flat=nn.flatten(enc_5)
            self.disc_output = nn.fully_connected(enc_5_flat, 1, activation_fn=tf.nn.sigmoid)


    def buildLoss(self):
        #Assign the variables to train to the two optimizer in order to allows separate training
        #of generator ad discriminator
        gen_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        dis_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")

        with tf.variable_scope('VAE_loss'):
            with tf.variable_scope('reconstruction_loss'):
                self.reconstr_loss=-tf.reduce_mean(tf.reduce_sum(self.gen_norm_x*tf.log(self.gen_output + 1e-9) + (1-self.gen_norm_x)*tf.log(1-self.gen_output + 1e-9), axis=[1,2,3]))

            with tf.variable_scope('KL_loss'):
                self.KLLoss= tf.reduce_mean(self.beta*(-0.5 * tf.reduce_sum(1.0 + tf.log(self.std +1e-9) - tf.square(self.mean) - self.std ,axis=-1)))

            self.vae_tot_loss = self.reconstr_loss + self.beta*self.KLLoss

            #called to train only the VAE (generator)
            self.vae_opt=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.vae_tot_loss, var_list=gen_vars)

        with tf.variable_scope('GAN_loss'):
            self.gan_error=-tf.reduce_mean(self.disc_Y*tf.log(self.disc_output + 1e-9) + (1-self.disc_Y)*tf.log(1- self.disc_output + 1e-9))  

            #called for training only discriminator on real data
            self.disc_opt=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.gan_error, var_list=dis_vars)
            #called for training all gan on fake data
            self.gan_opt=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.gan_error)

    def buildAccuracy(self):
        self.real_acc=tf.reduce_mean(tf.cast(self.disc_output<=0.1, tf.int32))
        self.fake_acc=tf.reduce_mean(tf.cast(self.disc_output>=0.9, tf.int32))

    def buildUtils(self):
        #Create file
        self.file=tf.summary.FileWriter(self.model_folder, self.sess.graph)

        with tf.name_scope('VAE_loss'):
            self.training_vae=tf.summary.merge([
                tf.summary.scalar('reconstruction_loss', self.reconstr_loss),
                tf.summary.scalar('KL_loss', self.KLLoss),
                tf.summary.scalar('total_loss', self.vae_tot_loss)
            ])

            self.testing_vae=tf.summary.merge([
                tf.summary.scalar('test_reconstruction_loss',self.reconstr_loss),
                tf.summary.scalar('test_KL_loss',self.KLLoss),
                tf.summary.scalar('test_total_loss',self.vae_tot_loss),
                tf.summary.image('real_images', self.gen_X),
                tf.summary.image('reconstruct_images', self.gen_output)
            ])

        firstGen=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator/encoder/Conv/weights")[0][0][0][0][0]
        lastGen=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator/decoder/Conv2d_transpose_5/weights")[0][0][0][0][0]
        firstDisc=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator/Conv/weights")[0][0][0][0][0]   
        lastDisc=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator/fully_connected/weights")[0][0][0]

        with tf.name_scope('GAN_loss_real_data'):
            #here add 4 gradients: last disc, first disc, last gen, first gen
            #only the discriminator should change
            self.training_discriminator_real=tf.summary.merge([
                tf.summary.scalar('discriminator_loss_real', self.gan_error),
                tf.summary.scalar('discriminator_lastlayer_w:0', lastDisc),
                tf.summary.scalar('discriminator_firstlayer_f:0', firstDisc),
                tf.summary.scalar('generator(NOCHANGE)_lastlayer_w:0', lastGen),
                tf.summary.scalar('generator(NOCHANGE)_firstlayer_f:0', firstGen)
            ])

            self.testing_discriminator_real=tf.summary.merge([
                tf.summary.scalar('discriminator_test_loss_real', self.real_acc)
            ])

        with tf.name_scope('GAN_loss_fake_data'):
            self.training_discriminator_fake=tf.summary.merge([
                tf.summary.scalar('discriminator_loss_fake', self.gan_error),
                tf.summary.scalar('discriminator_lastlayer_w:0', lastDisc),
                tf.summary.scalar('discriminator_firstlayer_f:0', firstDisc),
                tf.summary.scalar('generator_lastlayer_w:0', lastGen),
                tf.summary.scalar('generator_firstlayer_f:0', firstGen)
            ])

            self.testing_discriminator_fake=tf.summary.merge([
                tf.summary.scalar('discriminator_test_loss_fake', self.fake_acc),
                tf.summary.image('real_images', self.gen_X),
                tf.summary.image('reconstruct_images', self.gen_output)
            ])
        '''
        self.playing=tf.summary.merge([
            tf.summary.scalar('VAE_game_reconstruction_loss', self.reconstr_loss),
            tf.summary.scalar('VAE_game_KL_loss', self.KLLoss),
            tf.summary.scalar('VAE_game_total_loss', self.totLoss)
        ])
        '''

    def save(self):
        self.saver.save(self.sess, self.model_folder+"graph.ckpt")

    def encode(self, states):
        if (len(states.shape) == 3): #feeding single example
            states=np.expand_dims(states, axis=0)

        embeds=np.zeros((states.shape[0], self.latent_dim))
        for batchStart in range(0, states.shape[0], FLAGS.VAE_test_size):
            batchEnd=batchStart+FLAGS.VAE_test_size

            out=self.sess.run(self.latent, feed_dict={self.X: states[batchStart:batchEnd]})
            embeds[batchStart:batchEnd]=out
        return embeds
    
    def decode(self, embed):
        out=self.sess.run(self.gen_output, feed_dict={self.latent: embed})

        return out
