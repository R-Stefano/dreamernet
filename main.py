import matplotlib.pyplot as plt
import tensorflow as tf

from RNN import RNN
from VAE import VAE
from environment import Env
from trainer import Trainer
from utils import *

isVAETraining=True
isRNNTraining=True
def main():
    sess=tf.Session()

    #Initialize model
    vae = VAE(sess, isTraining=isVAETraining)
    rnn=RNN(sess, isTraining=isRNNTraining)

    #instantiate environment
    env=Env()

    #instantiate trainer
    trainer=Trainer()

    frames, actions=env.run()
    if(isVAETraining):
        trainer.trainVAE(sess, frames, vae)

    #the states must be processed by VAE
    embeddings=vae.predict(sess, frames)
    if(isRNNTraining):
        trainer.trainRNN(sess, embeddings, actions, rnn)


def visualizeEmbeddings(sess,env,frames, vae):
    #retrieve only 1000 frames
    frames=np.asarray(env.run()[:1000])

    e = sess.run(vae.latent, feed_dict={vae.X : frames/255.})

    visualizeEmbeddings(e, frames, "models/VAE/")


main()