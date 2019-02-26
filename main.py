import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 
import shutil #clean folder for retraining
from utils import preprocessingState
from models import RNN, VAEGAN, ACTOR, MCTS
from trainer import Trainer
from EnvWrapper import EnvWrap
import os
import preprocessing
import playing
test_envs={
    'frost':['FrostbiteNoFrameskip-v0', 18],
    'pong':['PongNoFrameskip-v0', 6]
}

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('preprocessing', False, 'If True, train the VAEGAN and the RNN before the agent')
flags.DEFINE_boolean('playing', True, 'If true, train the actor and the system on the games')

#ENVIRONMENT #env basic is 210,160,3
flags.DEFINE_integer('img_size', 96, 'dimension of the state to feed into the VAE')
flags.DEFINE_integer('crop_size', 160, 'dimension of the state after crop')
flags.DEFINE_integer('actions_size', 1, 'Number of actions in the environment. box2d is 3, ataray games is 1')
flags.DEFINE_integer('num_actions', 18, 'Number of possible actions in the environment if action is discreate')
flags.DEFINE_integer('gap', 28, 'How much crop from the top of the image')
flags.DEFINE_integer('init_frame_skip', 30, 'Number of frames to skip at the beginning of each game')
flags.DEFINE_integer('frame_skip', 4, 'Number of times an action is repeated')
flags.DEFINE_string('env', 'FrostbiteNoFrameskip-v0', 'The environment to use') # AirRaidNoFrameskip-v0 # #BreakoutNoFrameskip-v0  #CrazyClimber #JourneyEscape #Tutankham
flags.DEFINE_boolean('renderGame', False , 'Set to True to render the game')

#PREPROCESSING
flags.DEFINE_boolean('training_VAE', True, 'If True, start by training the VAE')
flags.DEFINE_boolean('training_VAEGAN', True, 'If True, train the VAEGAN model')
flags.DEFINE_boolean('testing_VAEGAN', False, 'If true testing the VAEGAN')

flags.DEFINE_boolean('training_RNN', True, 'If True, train the RNN model')
flags.DEFINE_boolean('testing_RNN', False, 'If true testing the RNN')

flags.DEFINE_integer('games', 5 , 'Number of times run the environment to create the data for preprocessing')
#VAEGAN HYPERPARAMS
flags.DEFINE_integer('VAEGAN_epoches', 1000, 'Number of times to repeat real-fake training')
flags.DEFINE_integer('VAEGAN_disc_real_epoches', 75, 'Number of epoches to train the discriminator on real data')
flags.DEFINE_integer('VAEGAN_disc_fake_epoches', 75, 'number of epoches to train the discriminator on fake data')
flags.DEFINE_integer('VAEGAN_gen_epoches', 50, 'number of epoches to train the discriminator on fake data')
flags.DEFINE_integer('VAE_training_epoches', 2000, 'Number of epoches to train VAE')
flags.DEFINE_integer('VAEGAN_train_size', 32, 'Number of frames to feed at each epoch')
flags.DEFINE_integer('VAEGAN_test_size', 64, 'Number of frames to feed at each epoch')

flags.DEFINE_boolean('use_only_GAN_loss', False, 'If true the error for the generator is only the abiity to fool the discriminator, else it is also added the VAE error')
flags.DEFINE_float('weight_VAE_loss', 0.5, 'If use_only_GAN_loss is False, this value decide the weight ot give to the VAE loss on the generator')
flags.DEFINE_integer('latent_dimension', 64, 'latent dimension')
flags.DEFINE_float('beta', 1, 'Disentangled Hyperparameter')

#RNN HYPERPARAMETERS
flags.DEFINE_integer('RNN_training_epoches', 2000, 'Number of epoches to train VAE')
flags.DEFINE_integer('RNN_train_size', 32, 'Number of frames to feed at each epoch')
flags.DEFINE_integer('RNN_test_size', 64, 'Number of frames to feed at each epoch')
flags.DEFINE_integer('sequence_length', 100, 'Total number of states to feed to the RNN')
flags.DEFINE_integer('hidden_units', 512, 'Number of hidden units in the LSTM layer')
flags.DEFINE_integer('LSTM_layers', 1, 'Number of the LSTM layers')
flags.DEFINE_integer('num_components', 5, 'Number of components of GMM')
flags.DEFINE_string('prediction_type', 'KL', 'The prediction can be MSE, GMM or KL')


#ACTOR
flags.DEFINE_integer('actor_training_games', 10000, 'Number of games to play while training the actor')
flags.DEFINE_integer('transition_buffer_size', 10000, 'Number of transitions to store')

flags.DEFINE_boolean('training_ACTOR', True, 'If True, train the ACTOR model')
flags.DEFINE_integer('ACTOR_input_size', 576, 'THe dimension of input vector')
flags.DEFINE_boolean('use_policy', False, 'If True use actor critic, otherwise value function')
flags.DEFINE_integer('actor_warmup', 5, 'If use MCTS is True, for the first N games, the actor not uses MCTS in order to get a good estimate first')


#MCTS
flags.DEFINE_integer('rollouts', 100, 'Number of rollouts at each timestep')
flags.DEFINE_boolean('use_MCTS', True, 'If true use the MCTS to decide which action execute')

if (FLAGS.preprocessing and (FLAGS.training_VAE and (len(os.listdir('models/VAEGAN/'))!=0) and FLAGS.training_VAEGAN)):
    print('cleaning VAE folder..')
    shutil.rmtree('models/VAEGAN/')#clean folder
if (FLAGS.preprocessing and (FLAGS.training_RNN and (len(os.listdir('models/RNN/'))!=0))):
    print('cleaning RNN folder..')
    shutil.rmtree('models/RNN/')#clean folder
'''
TODO:
keep selected node in MCTS for the next timestep
add terminal to the rnn prediction and so to the MCTS
maybe move from 4 frames to 8 frames skip (in order to have reaction time similar to human (0.15s))
(this implies retrain the RNN)
Implement prioritized experience replay

MAYBE TODO:
GMM for rnn
attention mechanism to the actor
add debugging to see if everything works and how the agent is behaving
'''
vae_sess=tf.Session()
rnn_sess=tf.Session()
actor_sess=tf.Session()

env=EnvWrap(FLAGS.init_frame_skip, FLAGS.frame_skip, FLAGS.env, FLAGS.renderGame)
vaegan=VAEGAN.VAEGAN(vae_sess)
rnn=RNN.RNN(rnn_sess)
actor=ACTOR.ACTOR(actor_sess)
mcts=MCTS.Tree(rnn, actor)
trainer=Trainer()

#If called, train the VAEGAN AND RNN before the actor
if (FLAGS.preprocessing):
    preprocessing.run(env, vaegan, trainer, rnn)

if (FLAGS.playing):
    #Make the actor play and train VAEGAN, RNN and actor
    playing.run(env, vaegan, rnn, actor, trainer, mcts)

'''
def main():
    
    #Tran alphazero using MCTS
    trainer.trainActor(mcts, vae, rnn, env, actor)
    
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