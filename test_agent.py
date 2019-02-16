import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 
import shutil #clean folder for retraining
from utils import preprocessingState

from models import RNN, VAE, ACTOR
from trainer import Trainer
from EnvWrapper import EnvWrap

flags = tf.app.flags
FLAGS = flags.FLAGS
#ENVIRONMENT
flags.DEFINE_integer('img_size', 96, 'dimension of the state to feed into the VAE')
flags.DEFINE_integer('crop_size', 160, 'dimension of the state after crop')
flags.DEFINE_integer('num_actions', 3, 'Number of possible actions in the environment')
flags.DEFINE_integer('gap', 35, 'How much crop from the top of the image')
flags.DEFINE_integer('init_frame_skip', 30, 'Number of frames to skip at the beginning of each game')
flags.DEFINE_integer('frame_skip', 4, 'Number of times an action is repeated')
flags.DEFINE_string('env', 'AssaultNoFrameskip-v0', 'The environment to use') #CarRacing-v0
flags.DEFINE_integer('games', 3 , 'Number of times run the environment to create the data')

#VAE
flags.DEFINE_boolean('training_VAE', False, 'If True, train the VAE model')
flags.DEFINE_boolean('testing_VAE', False, 'If true testing the VAE')
flags.DEFINE_integer('VAE_training_epoches', 2000, 'Number of epoches to train VAE')
flags.DEFINE_integer('VAE_train_size', 32, 'Number of frames to feed at each epoch')
flags.DEFINE_integer('VAE_test_size', 64, 'Number of frames to feed at each epoch')
#VAE HYPERPARAMETERS
flags.DEFINE_integer('latent_dimension', 64, 'latent dimension')
flags.DEFINE_float('beta', 1, 'Disentangled Hyperparameter')

#RNN
flags.DEFINE_boolean('training_RNN', False, 'If True, train the RNN model')
flags.DEFINE_boolean('testing_RNN', False, 'If true testing the RNN')
flags.DEFINE_integer('RNN_training_epoches', 2000, 'Number of epoches to train VAE')
flags.DEFINE_integer('RNN_train_size', 32, 'Number of frames to feed at each epoch')
flags.DEFINE_integer('RNN_test_size', 64, 'Number of frames to feed at each epoch')
#RNN HYPERPARAMETERS
flags.DEFINE_integer('sequence_length', 100, 'Total number of states to feed to the RNN')
flags.DEFINE_integer('hidden_units', 128, 'Number of hidden units in the LSTM layer')
flags.DEFINE_integer('LSTM_layers', 1, 'Number of the LSTM layers')

with tf.Session() as sess:
    env=EnvWrap(FLAGS.init_frame_skip, FLAGS.frame_skip, FLAGS.env)
    vae=VAE.VAE(sess)
    rnn=RNN.RNN(sess)
    actor=ACTOR.ACTOR(sess)
    trainer=Trainer()

    statesBuffer=[]
    actionsBuffer=[]
    rewardsBuffer=[]
    for i in range(1):
        env.initializeGame()

        while (not(d)):
            # quick state preprocessing
            s=preprocessingState(s)
            statesBuffer.append(s)

            #encode the state
            enc=vae.encode(s)
            print(enc.shape)
            #predict next state but retrieve the encoded version
            _,h=rnn.predict(enc)

            print('encoded state')
            print(enc)
            print('hidden state')
            print(h)

            #randomly sample action 0,1,2
            policy, value=actor.predict()
            print(policy.shape)
            actionsBuffer.append(a)

            s, r, d=env.repeatStep(a+1)
            rewardsBuffer.append(r)

            if d:
                statesBuffer.append(np.zeros((statesBuffer[-1].shape)))
                actionsBuffer.append(-1)