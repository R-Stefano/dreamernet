import matplotlib.pyplot as plt
import tensorflow as tf

from RNN import RNN
from VAE import VAE
from MCTS import Tree
from environment import Env
from trainer import Trainer
from utils import *

isVAETraining=False #True
isRNNTraining=False #True

rollouts=100
num_actions=3
def main():
    sess=tf.Session()

    #Initialize model
    vae = VAE(sess, isTraining=isVAETraining)
    rnn=RNN(sess, isTraining=isRNNTraining)
    mcts=Tree(num_actions, rollouts, rnn)

    #instantiate environment
    env=Env()

    #instantiate trainer
    trainer=Trainer()

    frames, actions=env.run()
    if(isVAETraining):
        trainer.trainVAE(sess, frames, vae)

    if(isRNNTraining):
        #the states must be processed by VAE
        embeddings=vae.predictBatch(sess, frames)
        trainer.trainRNN(sess, embeddings, actions, rnn)

    #Tran alphazero using MCTS
    trainer.trainAlphaZero(mcts, vae, rnn, env)

    
    '''
    ONce the VAE and RNN have been trained, I have to use the alphazero.
    Alphazero takes the current state and asks the MCTS to create the next states.
    No okay, so the next thhing is to give the current state to the MCTS that starts
    its algorithm. AT every iteration asks to alphazero to evaluate the node.

    Okay, now I designed the MCTS to output a single action, but I'm doing training.
    So, it should not stop to a single action but should iteratively play to train alphazero.

    Alphazero is trained increasing the policy confidence and trying to predict the reward.
    The problem is that at the moment I don't have any reward.

    Two solutions at the moment:
        1. make the RNN predicts also the reward (increasing the error and uncertainty, 
        so maybe destabilizing training)
        2. Train alphazero pseudo-online. It predicts actions until the end of the game.
        All predictions are stored. Then, it is trained on the real data.
    I would start with the second one.
    So, it takes the real state from the environment. It goes through the VAE and 
    retrieve the embedding. The embedding is going to be the root node of the MCTS.
    Here MCTS starts, it starts to expand the graph. It select an action which is used with 
    the embedding as input of the RNN. The result of the RNN creates the next node.
    '''

def visualizeEmbeddings(sess,env,frames, vae):
    #retrieve only 1000 frames
    frames=np.asarray(env.run()[:1000])

    e = sess.run(vae.latent, feed_dict={vae.X : frames/255.})

    visualizeEmbeddings(e, frames, "models/VAE/")


main()