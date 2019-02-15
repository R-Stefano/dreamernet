import tensorflow as tf
from models import VAE
from trainer import Trainer
from EnvWrapper import EnvWrap
import matplotlib.pyplot as plt
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('VAE_training_epoches', 2000, 'Number of epoches to train VAE')
flags.DEFINE_integer('VAE_train_size', 32, 'Number of frames to feed at each epoch')
flags.DEFINE_integer('VAE_test_size', 64, 'Number of frames to feed at each epoch')
flags.DEFINE_integer('latent_dimension', 32, 'latent dimension')
flags.DEFINE_boolean('training_VAE', True, 'If True, train the VAE model')
flags.DEFINE_float('beta', 1.4, 'Disentangled Hyperparameter')

flags.DEFINE_integer('img_size', 160, 'dimension of the state to feed into the VAE')
flags.DEFINE_integer('num_actions', 3, 'Number of possible actions in the environment')
flags.DEFINE_integer('gap', 35, 'How much crop from the top of the image')
flags.DEFINE_integer('init_frame_skip', 30, 'Number of frames to skip at the beginning of each game')
flags.DEFINE_integer('frame_skip', 4, 'Number of times an action is repeated')
flags.DEFINE_string('env', 'PongNoFrameskip-v4', 'The environment to use')

with tf.Session() as sess:
    env=EnvWrap(FLAGS.init_frame_skip, FLAGS.frame_skip, FLAGS.env)
    vae=VAE.VAE(sess)
    trainer=Trainer()

    frames, _=env.run(1)

    if(FLAGS.training_VAE):
        trainer.trainVAE(frames, vae)
    else:
        idxs=np.random.randint(0, frames.shape[0], 2)
        inputs=frames[idxs]
        '''
        out=vae.decode(vae.encode(inputs))
        out=(out*255).astype(int)
        
        f, axarr = plt.subplots(2,2)
        axarr[0,0].imshow(inputs[0])
        axarr[0,1].imshow(out[0])
        axarr[1,0].imshow(inputs[1])
        axarr[1,1].imshow(out[1])

        plt.show()
        '''

        plt.imshow(inputs[0]-inputs[1])
        plt.show()
