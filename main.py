import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
#Environment
flags.DEFINE_integer('num_actions', 3, 'Number of possible actions in the environment')
flags.DEFINE_integer('init_frame_skip', 30, 'Number of frames to skip at the beginning of each game')
flags.DEFINE_integer('frame_skip', 4, 'Number of times an action is repeated')
flags.DEFINE_string('env', 'PongNoFrameskip-v4', 'The environment to use')
flags.DEFINE_integer('simulation_epochs', 10, 'Number of games to play to generate the data to train VAE or RNN')

#Hyperparameters
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('latent_dimension', 32, 'Dimension of the embedded representation')
flags.DEFINE_boolean('image_preprocessing', True, 'If False, the image is fed into the VAE as it is')

#NOTE: CHECK WELL IN THE ENVIRONMENT CLASS. THE FINAL STATE IS ADDED AS A 64,64 MATRIX ALWAYS

#VAE 
flags.DEFINE_boolean('training_VAE', True, 'If True, train the VAE model')
flags.DEFINE_integer('VAE_training_epoches', 2000, 'Number of epoches to train VAE')
flags.DEFINE_integer('VAE_train_size', 32, 'Number of frames to feed at each epoch')
flags.DEFINE_integer('VAE_test_size', 64, 'Number of frames to feed at each epoch')

#RNN
flags.DEFINE_boolean('training_RNN', True, 'If True, train the RNN model')
flags.DEFINE_integer('sequence_length', 10, 'Total number of states to feed to the RNN')
flags.DEFINE_integer('hidden_units', 128, 'Number of hidden units in the LSTM layer')
flags.DEFINE_integer('RNN_training_epoches', 2000, 'Number of epoches to train VAE')
flags.DEFINE_integer('RNN_train_size', 32, 'Number of frames to feed at each epoch')
flags.DEFINE_integer('RNN_test_size', 64, 'Number of frames to feed at each epoch')
#MCTS
flags.DEFINE_integer('rollouts', 100, 'Number of simulations before selecting the action')

#ACTOR
flags.DEFINE_boolean('training_ACTOR', True, 'If True, train the ACTOR model')
flags.DEFINE_integer('actor_training_games', 10, 'Number of games to play to generate the transitions to train the Actor')
flags.DEFINE_integer('actor_training_steps', 200, 'Number of training steps after the training games')
flags.DEFINE_integer('actor_training_epochs', 100, 'Number of times repeate games + steps')
flags.DEFINE_integer('actor_testing_games', 10, 'Number of games to play after each training')

from models import VAE, RNN, ACTOR, MCTS
from EnvWrapper import EnvWrap
from trainer import Trainer
from utils import *
import matplotlib.pyplot as plt


def main():
    sess=tf.Session()

    #Instantiate the models
    vae = VAE.VAE(sess)
    rnn=RNN.RNN(sess)
    actor=ACTOR.ACTOR(sess)
    mcts=MCTS.Tree(rnn, actor)

    #instantiate environment
    env=EnvWrap(FLAGS.init_frame_skip, FLAGS.frame_skip, FLAGS.env, FLAGS.image_preprocessing)

    #instantiate trainer
    trainer=Trainer()

    if (FLAGS.training_VAE or FLAGS.training_RNN):
        #Generate samples to train VAE and RNN
        frames, actions=env.run(FLAGS.simulation_epochs)

    if(FLAGS.training_VAE):
        trainer.trainVAE(frames, vae)

    if(FLAGS.training_RNN):
        #the states must be processed by VAE
        embeddings=vae.encode(frames)
        trainer.trainRNN(embeddings, actions, rnn)
    
    #Tran alphazero using MCTS
    trainer.trainActor(mcts, vae, rnn, env, actor)
    
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