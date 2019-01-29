import matplotlib.pyplot as plt
import tensorflow as tf

from VAE import VAE
from environment import Env
from trainer import Trainer
from utils import *

def main():
    sess=tf.Session()

    #Initialize model
    vae = VAE()

    #instantiate environment
    env=Env()

    #instantiate trainer
    trainer=Trainer()

    frames=env.run()
    trainer.trainVAE(sess, frames, vae)


def visualizeEmbeddings(sess,env,frames, vae):
    #retrieve only 1000 frames
    frames=np.asarray(env.run()[:1000])

    e = sess.run(vae.latent, feed_dict={vae.X : frames/255.})

    visualizeEmbeddings(e, frames, "models/VAE/")


main()